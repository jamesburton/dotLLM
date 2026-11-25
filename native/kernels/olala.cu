// OVHcloud Olala CUDA kernels (#1984, epic #1969). Semantics: docs/OLALA.md; CPU reference: DotLLM.Cpu/Kernels/OlalaOps.cs and
// DotLLM.Models/Architectures/OlalaTransformerModel.cs.
//
// PRECISION CONTRACT (James, 2026-11-25; docs/OLALA.md section 4):
//   * Stored activations (anything that crosses a kernel boundary in global memory) are BF16 (act_t). Olala activations reach
//     ~1e9..1e14, so FP16 storage would overflow; F32 activation storage needs sign-off and is not shipped.
//   * Accumulation is in F32 registers (the geodesic reductions and the angle wrap use double, as on the CPU).
//   * The recurrent state (SSM matrix, rotary phase, trapezoid K/V, token-shift K/V) is F32 (OlalaStatePolicy).
//   * Weights are read quantized in place: Q8_0 / Q6_K / BF16 / F32 are dequantized in registers inside the dot loop. Never widened.
//   * Offsets are size_t: `long` is 32-bit under nvcc + MSVC (LLP64).
// Router logits / scores are never rounded to BF16: the router, sigmoid, bias, top-k and renormalisation are fused in one kernel.

#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

typedef unsigned short act_t;   // BF16 bits

__device__ __forceinline__ float bf2f(act_t v) { return __uint_as_float(((unsigned)v) << 16); }
__device__ __forceinline__ act_t f2bf(float f)
{
    unsigned b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (act_t)((b >> 16) | 0x40);   // NaN stays NaN
    b += 0x7fffu + ((b >> 16) & 1u);                                          // round to nearest even
    return (act_t)(b >> 16);
}
__device__ __forceinline__ float ldA(const act_t* p) { return bf2f(*p); }
__device__ __forceinline__ void stA(act_t* p, float v) { *p = f2bf(v); }

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : (x < -20.0f ? expf(x) : log1pf(expf(x))); }

__device__ __forceinline__ double blockSumD(double v, double* red /* [32] */)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    double s = 0.0;
    for (int i = 0; i < nw; ++i) s += red[i];
    return s;
}

// ── weight decoding: 8 weights per lane iteration ───────────────────────────────────────────────────────────────────────
#define WT_Q8_0 0
#define WT_Q6_K 1
#define WT_BF16 2
#define WT_F32  3

// Q8_0: struct { half d; int8 qs[32]; } = 34 bytes. Chunk c = 8 consecutive elements (4 chunks per block).
// BF16 / F32: chunk c = elements [8c, 8c + 8). Q6_K: one 256-element superblock per iteration, lane l owns elements
// {n*128 + l + 32k : n in 0..1, k in 0..3}.
template <int WT>
__device__ __forceinline__ void dq8(const uint8_t* row, int c, float (&w)[8])
{
    if (WT == WT_Q8_0)
    {
        const uint8_t* b = row + (size_t)(c >> 2) * 34;
        float d = __half2float(*reinterpret_cast<const __half*>(b));
        const signed char* q = reinterpret_cast<const signed char*>(b + 2) + (c & 3) * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = d * (float)q[j];
    }
    else if (WT == WT_BF16)
    {
        uint4 v = *reinterpret_cast<const uint4*>(row + (size_t)c * 16);
        w[0] = __uint_as_float(v.x << 16); w[1] = __uint_as_float(v.x & 0xffff0000u);
        w[2] = __uint_as_float(v.y << 16); w[3] = __uint_as_float(v.y & 0xffff0000u);
        w[4] = __uint_as_float(v.z << 16); w[5] = __uint_as_float(v.z & 0xffff0000u);
        w[6] = __uint_as_float(v.w << 16); w[7] = __uint_as_float(v.w & 0xffff0000u);
    }
    else // WT_F32
    {
        const float4* p = reinterpret_cast<const float4*>(row + (size_t)c * 32);
        float4 a = p[0], b = p[1];
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    }
}

__device__ __forceinline__ void dq_q6k(const uint8_t* row, int sb, int lane, float (&w)[8], int (&e)[8])
{
    const uint8_t* b = row + (size_t)sb * 210;
    float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
    const signed char* scales = reinterpret_cast<const signed char*>(b + 192);
    int isc = lane >> 4;
#pragma unroll
    for (int n = 0; n < 2; ++n)
    {
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const signed char* sc = scales + 8 * n;
        int qa = ql[lane], qb = ql[lane + 32], qc = qh[lane];
        int q1 = ((qa & 0xF) | (((qc >> 0) & 3) << 4)) - 32;
        int q2 = ((qb & 0xF) | (((qc >> 2) & 3) << 4)) - 32;
        int q3 = ((qa >> 4) | (((qc >> 4) & 3) << 4)) - 32;
        int q4 = ((qb >> 4) | (((qc >> 6) & 3) << 4)) - 32;
        w[n * 4 + 0] = d * (float)sc[isc] * (float)q1;
        w[n * 4 + 1] = d * (float)sc[isc + 2] * (float)q2;
        w[n * 4 + 2] = d * (float)sc[isc + 4] * (float)q3;
        w[n * 4 + 3] = d * (float)sc[isc + 6] * (float)q4;
        int base = sb * 256 + n * 128 + lane;
        e[n * 4 + 0] = base; e[n * 4 + 1] = base + 32; e[n * 4 + 2] = base + 64; e[n * 4 + 3] = base + 96;
    }
}

// One warp: dot of one weight row with TT activation rows (stride xStride elements), lane-partial sums (caller reduces).
template <int WT, int TT>
__device__ __forceinline__ void warp_dots(const uint8_t* row, const act_t* X, size_t xStride, int ntok, int In, int lane, float (&acc)[TT])
{
#pragma unroll
    for (int j = 0; j < TT; ++j) acc[j] = 0.0f;
    if (WT == WT_Q6_K)
    {
        int nsb = In >> 8;
        for (int sb = 0; sb < nsb; ++sb)
        {
            float w[8]; int e[8];
            dq_q6k(row, sb, lane, w, e);
#pragma unroll
            for (int j = 0; j < TT; ++j)
            {
                if (j < ntok)
                {
                    const act_t* x = X + (size_t)j * xStride;
                    float s = 0.0f;
#pragma unroll
                    for (int k = 0; k < 8; ++k) s += w[k] * ldA(x + e[k]);
                    acc[j] += s;
                }
            }
        }
    }
    else
    {
        int nch = In >> 3;
        for (int c = lane; c < nch; c += 32)
        {
            float w[8];
            dq8<WT>(row, c, w);
            int e0 = c * 8;
#pragma unroll
            for (int j = 0; j < TT; ++j)
            {
                if (j < ntok)
                {
                    const act_t* x = X + (size_t)j * xStride + e0;
                    float s = 0.0f;
#pragma unroll
                    for (int k = 0; k < 8; ++k) s += w[k] * ldA(x + k);
                    acc[j] += s;
                }
            }
        }
    }
}

// Dense projection Y[t][row] = W[row] . X[t] for TT tokens per block row. epi: 0 store, 1 ReLU^2, 2 add aux[t][row] (BF16).
// outF32: store the F32 accumulator (the logits only) instead of BF16.
template <int WT, int TT>
__device__ __forceinline__ void gemv_impl(const uint8_t* W, size_t rowBytes, const act_t* X, void* Y, int In, int Out, int ntok,
                                          int outF32, int epi, const act_t* aux)
{
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (row >= Out) return;
    int t0 = blockIdx.y * TT;
    int nt = min(TT, ntok - t0);
    float acc[TT];
    warp_dots<WT, TT>(W + (size_t)row * rowBytes, X + (size_t)t0 * In, (size_t)In, nt, In, lane, acc);
#pragma unroll
    for (int j = 0; j < TT; ++j) acc[j] = warpSum(acc[j]);
    if (lane == 0)
    {
#pragma unroll
        for (int j = 0; j < TT; ++j)
        {
            if (j < nt)
            {
                float v = acc[j];
                size_t o = (size_t)(t0 + j) * Out + row;
                if (epi == 1) { v = fmaxf(v, 0.0f); v *= v; }
                else if (epi == 2) v += ldA(aux + o);
                if (outF32) reinterpret_cast<float*>(Y)[o] = v; else stA(reinterpret_cast<act_t*>(Y) + o, v);
            }
        }
    }
}

#define GEMV_KERNEL(NAME, WT, TT)                                                                                              \
    extern "C" __global__ void __launch_bounds__(256) NAME(const uint8_t* __restrict__ W, size_t rowBytes,                      \
        const act_t* __restrict__ X, void* __restrict__ Y, int In, int Out, int ntok, int outF32, int epi,                     \
        const act_t* __restrict__ aux)                                                                                          \
    { gemv_impl<WT, TT>(W, rowBytes, X, Y, In, Out, ntok, outF32, epi, aux); }

GEMV_KERNEL(olala_gemv_q8_0_t1, WT_Q8_0, 1)
GEMV_KERNEL(olala_gemv_q8_0_t8, WT_Q8_0, 8)
GEMV_KERNEL(olala_gemv_q6_k_t1, WT_Q6_K, 1)
GEMV_KERNEL(olala_gemv_q6_k_t8, WT_Q6_K, 8)
GEMV_KERNEL(olala_gemv_bf16_t1, WT_BF16, 1)
GEMV_KERNEL(olala_gemv_bf16_t8, WT_BF16, 8)
GEMV_KERNEL(olala_gemv_f32_t1, WT_F32, 1)
GEMV_KERNEL(olala_gemv_f32_t8, WT_F32, 8)

// ── routed experts (one expert bank per layer, experts stacked) ─────────────────────────────────────────────────────────
// Up: Y[a][row] = relu(W[e_a][row] . X[token(a)])^2, a = token * K + slot. grid (ceil(Out / 8), T * K).
template <int WT>
__device__ __forceinline__ void moe_up_impl(const uint8_t* W, size_t expertBytes, size_t rowBytes, const int* idx, const act_t* X,
                                            act_t* Y, int In, int Out, int K)
{
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (row >= Out) return;
    int a = blockIdx.y, t = a / K;
    size_t e = (size_t)idx[a];
    float acc[1];
    warp_dots<WT, 1>(W + e * expertBytes + (size_t)row * rowBytes, X + (size_t)t * In, (size_t)In, 1, In, lane, acc);
    float v = warpSum(acc[0]);
    if (lane == 0) { v = fmaxf(v, 0.0f); stA(Y + (size_t)a * Out + row, v * v); }
}

// Down: Y[t][row] = sum_s w[t][s] * (W[e_ts][row] . Hid[t * K + s]). grid (ceil(Out / 8), T).
template <int WT>
__device__ __forceinline__ void moe_down_impl(const uint8_t* W, size_t expertBytes, size_t rowBytes, const int* idx, const float* wts,
                                              const act_t* Hid, act_t* Y, int In, int Out, int K)
{
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (row >= Out) return;
    int t = blockIdx.y;
    float total = 0.0f;
    for (int s = 0; s < K; ++s)
    {
        size_t a = (size_t)t * K + s;
        size_t e = (size_t)idx[a];
        float acc[1];
        warp_dots<WT, 1>(W + e * expertBytes + (size_t)row * rowBytes, Hid + a * In, (size_t)In, 1, In, lane, acc);
        total += wts[a] * warpSum(acc[0]);
    }
    if (lane == 0) stA(Y + (size_t)t * Out + row, total);
}

#define MOE_KERNELS(SUFFIX, WT)                                                                                                \
    extern "C" __global__ void __launch_bounds__(256) olala_moe_up_##SUFFIX(const uint8_t* __restrict__ W, size_t expertBytes,   \
        size_t rowBytes, const int* __restrict__ idx, const act_t* __restrict__ X, act_t* __restrict__ Y, int In, int Out, int K) \
    { moe_up_impl<WT>(W, expertBytes, rowBytes, idx, X, Y, In, Out, K); }                                                       \
    extern "C" __global__ void __launch_bounds__(256) olala_moe_down_##SUFFIX(const uint8_t* __restrict__ W, size_t expertBytes, \
        size_t rowBytes, const int* __restrict__ idx, const float* __restrict__ wts, const act_t* __restrict__ Hid,             \
        act_t* __restrict__ Y, int In, int Out, int K)                                                                          \
    { moe_down_impl<WT>(W, expertBytes, rowBytes, idx, wts, Hid, Y, In, Out, K); }

MOE_KERNELS(q8_0, WT_Q8_0)
MOE_KERNELS(q6_k, WT_Q6_K)
MOE_KERNELS(bf16, WT_BF16)
MOE_KERNELS(f32, WT_F32)

// ── router: logits (F32 weights) -> sigmoid -> top-K of (score + bias) -> renormalised x scale. One block per token. ─────
// Scores are never rounded to a narrow type: near-tied experts (margins ~1e-5) must resolve exactly as in F32.
// Dynamic smem: (H + 2 * E) floats.
extern "C" __global__ void __launch_bounds__(256) olala_route(
    const float* __restrict__ Wr, const float* __restrict__ bias, const act_t* __restrict__ X, int* __restrict__ idxOut,
    float* __restrict__ wOut, int H, int E, int K, float scale)
{
    extern __shared__ float sm[];
    float* xs = sm;            // [H]
    float* sig = xs + H;       // [E]
    float* sc = sig + E;       // [E] sigmoid + bias; selected entries are overwritten with -inf
    int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = ldA(X + (size_t)t * H + i);
    __syncthreads();
    for (int e = warp; e < E; e += 8)
    {
        const float* r = Wr + (size_t)e * H;
        float s = 0.0f;
        for (int i = lane; i < H; i += 32) s += r[i] * xs[i];
        s = warpSum(s);
        if (lane == 0) { float g = sigmoidf_(s); sig[e] = g; sc[e] = g + bias[e]; }
    }
    __syncthreads();
    if (warp == 0)
    {
        float chosenSum = 0.0f;
        float cw[16]; int ci[16];   // K <= 16
        for (int k = 0; k < K; ++k)
        {
            float best = -INFINITY; int bi = 0x7fffffff;
            for (int e = lane; e < E; e += 32)
            {
                float v = sc[e];
                if (v > best || (v == best && e < bi)) { best = v; bi = e; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                float ov = __shfl_xor_sync(0xffffffffu, best, o);
                int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            if (k < 16) { ci[k] = bi; cw[k] = sig[bi]; }
            chosenSum += sig[bi];
            __syncwarp();
            if (lane == 0) sc[bi] = -INFINITY;
            __syncwarp();
        }
        if (lane == 0)
            for (int k = 0; k < K; ++k)
            {
                idxOut[(size_t)t * K + k] = ci[k];
                wOut[(size_t)t * K + k] = cw[k] / (chosenSum + 1e-20f) * scale;
            }
    }
}

// ── geodesic residual (OlalaGeodesicNorm): x <- x cos(theta) + unit(g_perp) |x| sin(theta). Double reductions (g reaches ~1e14). ─
extern "C" __global__ void __launch_bounds__(256) olala_geodesic(act_t* __restrict__ x, const act_t* __restrict__ g, int H, float scale,
                                                                 float bias, int layer)
{
    __shared__ double red[32];
    size_t base = (size_t)blockIdx.x * H;
    double xx = 0.0, xg = 0.0;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
    {
        double xv = ldA(x + base + i), gv = ldA(g + base + i);
        xx += xv * xv; xg += xv * gv;
    }
    xx = blockSumD(xx, red);
    xg = blockSumD(xg, red);
    double xNormSq = fmax(xx, 1e-12);
    double proj = xg / xNormSq;
    double tt = 0.0;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
    {
        double d = (double)ldA(g + base + i) - proj * (double)ldA(x + base + i);
        tt += d * d;
    }
    tt = blockSumD(tt, red);
    double tangentNorm = sqrt(tt);
    double safeTangent = fmax(tangentNorm, 1e-8);
    double r = sqrt(xx);
    double safeR = fmax(r, 1e-6);
    const double clampV = (double)0.78539816339744830962f;   // MathF.PI / 4f, as the CPU
    double theta = fmin(safeTangent / safeR, clampV);
    theta = fmin((theta * (double)scale + (double)bias) / (double)(layer + 1), clampV);
    double c = cos(theta), s = sin(theta);
    double coeff = safeR * s / safeTangent;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
    {
        double xv = ldA(x + base + i);
        double d = (double)ldA(g + base + i) - proj * xv;
        stA(x + base + i, (float)(xv * c + d * coeff));
    }
}

// ── embedding lookup + L2 normalisation (normalize_embeddings_ngpt). Dynamic smem: H floats. ──────────────────────────────
template <int WT>
__device__ __forceinline__ void embed_impl(const uint8_t* W, size_t rowBytes, const int* ids, act_t* out, int H)
{
    extern __shared__ float srow[];
    __shared__ double red[32];
    const uint8_t* row = W + (size_t)ids[blockIdx.x] * rowBytes;
    double ss = 0.0;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
    {
        float v;
        if (WT == WT_Q8_0)
        {
            const uint8_t* b = row + (size_t)(i >> 5) * 34;
            v = __half2float(*reinterpret_cast<const __half*>(b)) * (float)reinterpret_cast<const signed char*>(b + 2)[i & 31];
        }
        else if (WT == WT_BF16) v = bf2f(reinterpret_cast<const act_t*>(row)[i]);
        else v = reinterpret_cast<const float*>(row)[i];
        srow[i] = v;
        ss += (double)v * (double)v;
    }
    ss = blockSumD(ss, red);
    float norm = fmaxf((float)sqrt(ss), 1e-12f);
    for (int i = threadIdx.x; i < H; i += blockDim.x) stA(out + (size_t)blockIdx.x * H + i, srow[i] / norm);
}

#define EMBED_KERNEL(NAME, WT)                                                                                                 \
    extern "C" __global__ void __launch_bounds__(256) NAME(const uint8_t* __restrict__ W, size_t rowBytes,                      \
        const int* __restrict__ ids, act_t* __restrict__ out, int H)                                                            \
    { embed_impl<WT>(W, rowBytes, ids, out, H); }

EMBED_KERNEL(olala_embed_q8_0, WT_Q8_0)
EMBED_KERNEL(olala_embed_bf16, WT_BF16)
EMBED_KERNEL(olala_embed_f32, WT_F32)

// ── Mamba-3 MIMO step, T tokens sequentially, one block (16 warps) per head. ───────────────────────────────────────────────
// Per token (OlalaOps.Mamba3MimoStep): B/C RMSNorm (+bias, data-dependent rotation by the cumulative angle), then per P row
//   h = alpha h + sum_r (xv_r gamma k_r + xs_r beta kPrev_r);  o_r = h . q_r;  y = sum_r (o_r + xv_r D) silu(z mz_r) mo_r.
// The SSM matrix row lives in registers across the T tokens: lane n-slice {lane + 32 j}, rows {warp + 16 i}.
// Limits (validated on the host): N <= 128, P <= 64, R <= 4, A <= 64, A <= N/2.
// Dynamic smem: (3 * R * N + 3 * A) floats.
#define MB_NL 4
#define MB_RW 4
extern "C" __global__ void __launch_bounds__(512) olala_mamba3_mimo(
    const act_t* __restrict__ zxdt, const act_t* __restrict__ dyn, int T,
    float* __restrict__ ssm, float* __restrict__ angle, float* __restrict__ kst, float* __restrict__ vst,
    const float* __restrict__ bnorm, const float* __restrict__ cnorm, const float* __restrict__ bbias, const float* __restrict__ cbias,
    const float* __restrict__ dtbias, const float* __restrict__ dskip, const float* __restrict__ mimoX,
    const float* __restrict__ mimoZ, const float* __restrict__ mimoO, act_t* __restrict__ y,
    int H, int P, int N, int R, int A, float eps)
{
    extern __shared__ float sm[];
    float* kS = sm;                 // [R*N] rotated K (this token)
    float* qS = kS + R * N;         // [R*N] rotated Q
    float* kPrev = qS + R * N;      // [R*N] previous token's rotated K
    float* cosS = kPrev + R * N;    // [A]
    float* sinS = cosS + A;         // [A]
    float* angS = sinS + A;         // [A]
    const int h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ph = 2 * P + 3, dynW = 2 * R * N + A, half = N >> 1;

    for (int i = tid; i < R * N; i += blockDim.x) kPrev[i] = kst[((size_t)(i / N) * H + h) * N + (i % N)];
    for (int a = tid; a < A; a += blockDim.x) angS[a] = angle[(size_t)h * A + a];
    float s[MB_RW][MB_NL];
    float vprev[MB_RW];
#pragma unroll
    for (int i = 0; i < MB_RW; ++i)
    {
        int p = warp + 16 * i;
        vprev[i] = (p < P) ? vst[(size_t)h * P + p] : 0.0f;
#pragma unroll
        for (int j = 0; j < MB_NL; ++j)
        {
            int n = lane + 32 * j;
            s[i][j] = (p < P && n < N) ? ssm[((size_t)h * P + p) * N + n] : 0.0f;
        }
    }
    __syncthreads();

    for (int t = 0; t < T; ++t)
    {
        const act_t* zrow = zxdt + (size_t)t * H * ph + (size_t)h * ph;
        const act_t* drow = dyn + (size_t)t * dynW;
        float dtRaw = ldA(zrow + 2 * P), aRaw = ldA(zrow + 2 * P + 1), trapRaw = ldA(zrow + 2 * P + 2);
        float decay = fminf(-softplusf_(aRaw), -1e-4f);
        float dt = softplusf_(dtRaw + dtbias[h]);
        float trap = sigmoidf_(trapRaw);
        float alpha = expf(decay * dt);
        float beta = (1.0f - trap) * dt * alpha;
        float gamma = trap * dt;

        // phase 1: B / C RMSNorm rows (warps 0..2R-1) and the cumulative angle (warp 15)
        if (warp < 2 * R)
        {
            int r = warp < R ? warp : warp - R;
            bool isB = warp < R;
            const act_t* src = drow + (isB ? 0 : R * N) + r * N;
            float v[MB_NL], ss = 0.0f;
#pragma unroll
            for (int j = 0; j < MB_NL; ++j)
            {
                int n = lane + 32 * j;
                v[j] = n < N ? ldA(src + n) : 0.0f;
                ss += v[j] * v[j];
            }
            ss = warpSum(ss);
            float rms = sqrtf(ss / (float)N + eps);
            float scl = 1.0f / rms;
            const float* gain = isB ? bnorm : cnorm;
            const float* bia = (isB ? bbias : cbias) + ((size_t)h * R + r) * N;
            float* dst = (isB ? kS : qS) + r * N;
#pragma unroll
            for (int j = 0; j < MB_NL; ++j)
            {
                int n = lane + 32 * j;
                if (n < N) dst[n] = v[j] * scl * gain[n] + bia[n];
            }
        }
        else if (warp == 15)
        {
            const double TWO_PI = 6.283185307179586476925286766559;
            for (int a = lane; a < A; a += 32)
            {
                double v = (double)angS[a] + (double)tanhf(ldA(drow + 2 * R * N + a)) * (double)dt * 3.1415926535897932384626433832795;
                v -= floor(v / TWO_PI) * TWO_PI;
                float vf = (float)v;
                angS[a] = vf;
                cosS[a] = cosf(vf);
                sinS[a] = sinf(vf);
            }
        }
        __syncthreads();
        // phase 2: rotate pairs (a, N/2 + a) of every rank row
        for (int i = tid; i < R * A; i += blockDim.x)
        {
            int r = i / A, a = i % A;
            int i0 = r * N + a, i1 = r * N + half + a;
            float c = cosS[a], sn = sinS[a];
            float q0 = qS[i0], q1 = qS[i1], k0 = kS[i0], k1 = kS[i1];
            qS[i0] = q0 * c - q1 * sn; qS[i1] = q0 * sn + q1 * c;
            kS[i0] = k0 * c - k1 * sn; kS[i1] = k0 * sn + k1 * c;
        }
        __syncthreads();
        // phase 3: state update + readout, rows {warp + 16 i}
        float dsk = dskip[h];
#pragma unroll
        for (int i = 0; i < MB_RW; ++i)
        {
            int p = warp + 16 * i;
            if (p >= P) continue;
            float xp = ldA(zrow + P + p), zp = ldA(zrow + p);
            float xv[4], xs[4], o[4];
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                if (r < R)
                {
                    float mx = mimoX[((size_t)h * R + r) * P + p];
                    xv[r] = xp * mx; xs[r] = vprev[i] * mx;
                }
                else { xv[r] = 0.0f; xs[r] = 0.0f; }
                o[r] = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < MB_NL; ++j)
            {
                int n = lane + 32 * j;
                if (n < N)
                {
                    float sv = s[i][j] * alpha;
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        if (r < R) sv += xv[r] * gamma * kS[r * N + n] + xs[r] * beta * kPrev[r * N + n];
                    s[i][j] = sv;
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        if (r < R) o[r] += sv * qS[r * N + n];
                }
            }
            float acc = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                float orr = warpSum(o[r]);
                if (r < R)
                {
                    float mz = mimoZ[((size_t)h * R + r) * P + p];
                    float mo = mimoO[((size_t)h * R + r) * P + p];
                    acc += (orr + xv[r] * dsk) * siluf_(zp * mz) * mo;
                }
            }
            if (lane == 0) stA(y + (size_t)t * H * P + (size_t)h * P + p, acc);
            vprev[i] = xp;
        }
        __syncthreads();
        for (int i = tid; i < R * N; i += blockDim.x) kPrev[i] = kS[i];
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MB_RW; ++i)
    {
        int p = warp + 16 * i;
        if (p >= P) continue;
        if (lane == 0) vst[(size_t)h * P + p] = vprev[i];
#pragma unroll
        for (int j = 0; j < MB_NL; ++j)
        {
            int n = lane + 32 * j;
            if (n < N) ssm[((size_t)h * P + p) * N + n] = s[i][j];
        }
    }
    for (int i = tid; i < R * N; i += blockDim.x) kst[((size_t)(i / N) * H + h) * N + (i % N)] = kPrev[i];
    for (int a = tid; a < A; a += blockDim.x) angle[(size_t)h * A + a] = angS[a];
}

// ── differential TPA attention ────────────────────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float tpa_raw(const act_t* a, const act_t* b, int t, int nk, int h, int d, int rank, int hd)
{
    float v = 0.0f;
    for (int r = 0; r < rank; ++r)
        v += ldA(a + (size_t)t * nk * rank + h * rank + r) * ldA(b + (size_t)t * rank * hd + r * hd + d);
    return v * (1.0f / (float)rank);
}

// K/V of T new tokens: TPA reconstruction, token shift against the previous token's RAW K/V (F32 state for t == 0), K RMSNorm, BF16 cache
// write at positions posBase + t. grid (nk, T), block 128 (hd <= 128). Reads lastK/lastV only; tpa_carry updates them afterwards.
extern "C" __global__ void __launch_bounds__(128) olala_tpa_kv(
    const act_t* __restrict__ ak, const act_t* __restrict__ av, const act_t* __restrict__ bk, const act_t* __restrict__ bv,
    const act_t* __restrict__ sk, const act_t* __restrict__ sv, const float* __restrict__ lastK, const float* __restrict__ lastV,
    act_t* __restrict__ kcache, act_t* __restrict__ vcache, const float* __restrict__ kgain,
    int posBase, int T, int nk, int rank, int hd, float eps)
{
    __shared__ float red[4];
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    int pos = posBase + t;
    size_t kvW = (size_t)nk * hd;
    bool act = d < hd;
    float rawK = 0.0f, rawV = 0.0f, prevK = 0.0f, prevV = 0.0f;
    if (act)
    {
        rawK = tpa_raw(ak, bk, t, nk, h, d, rank, hd);
        rawV = tpa_raw(av, bv, t, nk, h, d, rank, hd);
        if (pos > 0)
        {
            prevK = t > 0 ? tpa_raw(ak, bk, t - 1, nk, h, d, rank, hd) : lastK[h * hd + d];
            prevV = t > 0 ? tpa_raw(av, bv, t - 1, nk, h, d, rank, hd) : lastV[h * hd + d];
        }
    }
    float alK = pos == 0 ? 0.0f : sigmoidf_(ldA(sk + (size_t)t * nk + h));
    float alV = pos == 0 ? 0.0f : sigmoidf_(ldA(sv + (size_t)t * nk + h));
    float kk = alK * prevK + (1.0f - alK) * rawK;
    float vv = alV * prevV + (1.0f - alV) * rawV;
    float ss = act ? kk * kk : 0.0f;
    ss = warpSum(ss);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    float tot = red[0] + red[1] + red[2] + red[3];
    float rms = sqrtf(tot / (float)hd + eps);
    float scl = 1.0f / rms;
    if (act)
    {
        stA(kcache + (size_t)pos * kvW + (size_t)h * hd + d, kk * scl * kgain[d]);
        stA(vcache + (size_t)pos * kvW + (size_t)h * hd + d, vv);
    }
}

// Token-shift carry: the last token's RAW (pre-shift, pre-norm) K and V, stored F32 (OlalaStatePolicy.TokenShift). grid nk, block 128.
extern "C" __global__ void __launch_bounds__(128) olala_tpa_carry(
    const act_t* __restrict__ ak, const act_t* __restrict__ av, const act_t* __restrict__ bk, const act_t* __restrict__ bv,
    float* __restrict__ lastK, float* __restrict__ lastV, int T, int nk, int rank, int hd)
{
    int h = blockIdx.x, d = threadIdx.x;
    if (d >= hd) return;
    lastK[h * hd + d] = tpa_raw(ak, bk, T - 1, nk, h, d, rank, hd);
    lastV[h * hd + d] = tpa_raw(av, bv, T - 1, nk, h, d, rank, hd);
}

// Attention for T new tokens over the BF16 K/V history. Query: RMSNorm * gain * scaler[h] * log(min(pos + 1, window)) in F32 registers;
// scores = q.k / head_dim, soft cap, causal window of window + 1 keys; online softmax over 8 warps. grid (nq, T), block 256, hd <= 128.
extern "C" __global__ void __launch_bounds__(256) olala_attn(
    const act_t* __restrict__ Q, const act_t* __restrict__ Kc, const act_t* __restrict__ Vc, act_t* __restrict__ out,
    const float* __restrict__ qgain, const float* __restrict__ scaler,
    int posBase, int nq, int nk, int hd, int window, float softcap, float eps)
{
    __shared__ float smM[8], smL[8];
    __shared__ float smAcc[8][128];
    int h = blockIdx.x, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int pos = posBase + t;
    int group = nq / nk, kvh = h / group;
    size_t kvW = (size_t)nk * hd, qW = (size_t)nq * hd;
    float q[4], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int d = lane + 32 * j;
        float v = d < hd ? ldA(Q + (size_t)t * qW + (size_t)h * hd + d) : 0.0f;
        q[j] = v; ss += v * v;
    }
    ss = warpSum(ss);
    float scl = 1.0f / sqrtf(ss / (float)hd + eps);
    float logPos = logf(fminf((float)(pos + 1), (float)window));
    float qs = scaler[h] * logPos;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int d = lane + 32 * j;
        q[j] = d < hd ? (q[j] * scl * qgain[d]) * qs : 0.0f;
    }
    int lo = max(0, pos - window);
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float invHd = 1.0f / (float)hd;
    for (int j = lo + warp; j <= pos; j += 8)
    {
        const act_t* kp = Kc + (size_t)j * kvW + (size_t)kvh * hd;
        const act_t* vp = Vc + (size_t)j * kvW + (size_t)kvh * hd;
        float s = 0.0f;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
        {
            int d = lane + 32 * jj;
            if (d < hd) s += q[jj] * ldA(kp + d);
        }
        s = warpSum(s) * invHd;
        if (softcap > 0.0f) s = tanhf(s / softcap) * softcap;
        float mn = fmaxf(m, s);
        float corr = (m == -INFINITY) ? 0.0f : expf(m - mn);
        float pr = expf(s - mn);
        l = l * corr + pr;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
        {
            int d = lane + 32 * jj;
            acc[jj] = acc[jj] * corr + (d < hd ? pr * ldA(vp + d) : 0.0f);
        }
        m = mn;
    }
    if (lane == 0) { smM[warp] = m; smL[warp] = l; }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj)
    {
        int d = lane + 32 * jj;
        if (d < hd) smAcc[warp][d] = acc[jj];
    }
    __syncthreads();
    if (threadIdx.x < hd)
    {
        float M = -INFINITY;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, smM[w]);
        float L = 0.0f, o = 0.0f;
        for (int w = 0; w < 8; ++w)
        {
            if (smM[w] == -INFINITY) continue;
            float f = expf(smM[w] - M);
            L += smL[w] * f; o += smAcc[w][threadIdx.x] * f;
        }
        stA(out + (size_t)t * qW + (size_t)h * hd + threadIdx.x, o / L);
    }
}

// Differential combine + output gate: out[g*snr + s][d] = (attn[g*group + s] - sigmoid(lambda[g]) * attn[g*group + snr]) * silu(gate + gateBias).
// grid T, block 256.
extern "C" __global__ void __launch_bounds__(256) olala_attn_combine(
    const act_t* __restrict__ attn, const act_t* __restrict__ lam, const act_t* __restrict__ gate, act_t* __restrict__ out,
    int nk, int group, int hd, float gateBias)
{
    int t = blockIdx.x;
    int snr = group - 1;
    size_t qW = (size_t)nk * group * hd;
    size_t gateW = (size_t)nk * snr * hd;
    for (int i = threadIdx.x; i < (int)gateW; i += blockDim.x)
    {
        int g = i / (snr * hd), rem = i % (snr * hd);
        int s = rem / hd, d = rem % hd;
        float lambda = sigmoidf_(ldA(lam + (size_t)t * nk + g));
        float a = ldA(attn + (size_t)t * qW + (size_t)(g * group + s) * hd + d);
        float nz = ldA(attn + (size_t)t * qW + (size_t)(g * group + snr) * hd + d);
        float gt = siluf_(ldA(gate + (size_t)t * gateW + i) + gateBias);
        stA(out + (size_t)t * gateW + i, (a - lambda * nz) * gt);
    }
}
