// Ternary residual adapter (.nsadapter), in place on the F32 residual stream (#1618).
//
// CUDA port of native/vulkan/shaders/ternary_adapter_residual_f32.comp, which reproduces the CPU reference
// (src/DotLLM.Cpu/Kernels/TernaryAdapterKernel.cs) bit for bit:
//
//   x8   = absmax_int8(h)                  scale 127/max|h|, round-half-even, clamp [-128,127]
//   d_j  = float(sum_c Wdown[j,c] * x8[c]) * downScale[j];  r_j = max(0, d_j);  m = max_j r_j   (per circuit)
//   open = alpha > 0 && m > max(threshold, 1e-6)
//   q_j  = clamp(rint(r_j * (127/m)), -128, 127)
//   u_d  = float(sum_j Wup[d,j] * q_j) * upScale[d];   dq = (m/127) * alpha
//   h'_d = h_d + (0 + u1_d*dq1 + u2_d*dq2 + ...)       (circuits of one layer in registration order)
//
// Weights are the CPU hook's in-place plus/minus bitmask layout (2 bits/weight): per mask row, ceil(d/64) blocks of
// [plus64, minus64]; `masks` holds totalRank Down rows then totalRank UpT rows (UpT row j = column j of W_up). Nothing is
// expanded. All float ops are explicit IEEE round-to-nearest intrinsics (no FMA contraction, correctly rounded division).
// One thread block per token row, 256 threads; summed rank <= 256, d <= 8192.
//
// Validated on an RTX 3060 (bit-exact vs the CPU hook, #1625). The .ptx must be built with the CUDA 12.8 toolchain
// (.version 8.7) via native/build_ptx.bat and committed as native/ptx/ternary_adapter_residual_f32.ptx; the MSBuild compile of this kernel is non-fatal.

#include <cuda_fp16.h>

#define TA_THREADS 256
#define TA_MAX_D 8192

struct TaCircuitMeta
{
    int rank;
    int rankOffset;   // first row of this circuit in the concatenated rank space
    float alpha;
    float threshold;
};

// Correctly rounded a / b (positive finite): __fdiv_rn is IEEE; kept as a named helper to mirror the Vulkan divRN.
__device__ __forceinline__ float ta_div_rn(float a, float b) { return __fdiv_rn(a, b); }

template <bool SCALE_F16>
__device__ __forceinline__ float ta_scale(const void* scales, int idx)
{
    if (SCALE_F16) return __half2float(((const __half*)scales)[idx]);
    return ((const float*)scales)[idx];
}

template <bool SCALE_F16>
__device__ __forceinline__ void ternary_adapter_body(
    float* __restrict__ h, const unsigned long long* __restrict__ masks, const void* __restrict__ scales,
    const TaCircuitMeta* __restrict__ meta, int d, int nRows, int nCirc, int totalRank, int wprU64,
    int upScaleBase, int upScaleStride)
{
    __shared__ signed char x8[TA_MAX_D];
    __shared__ float rj[TA_THREADS];
    __shared__ int qv[TA_THREADS];
    __shared__ float dqv[TA_THREADS];
    __shared__ int openv[TA_THREADS];
    __shared__ float redF[TA_THREADS];
    __shared__ float shScale;

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= nRows) return;
    float* row = h + (size_t)t * d;

    // ---- A: absmax -> int8 row ----
    float mx = 0.0f;
    for (int i = tid; i < d; i += TA_THREADS) mx = fmaxf(mx, fabsf(row[i]));
    redF[tid] = mx;
    __syncthreads();
    for (int s = TA_THREADS / 2; s > 0; s >>= 1)
    {
        if (tid < s) redF[tid] = fmaxf(redF[tid], redF[tid + s]);
        __syncthreads();
    }
    if (tid == 0) shScale = (redF[0] > 1e-6f) ? ta_div_rn(127.0f, redF[0]) : 0.0f;
    __syncthreads();
    const float qsx = shScale;
    for (int i = tid; i < d; i += TA_THREADS)
    {
        int q = (int)rintf(__fmul_rn(row[i], qsx));          // rintf = round-half-even (default rounding mode)
        x8[i] = (signed char)max(-128, min(127, q));
    }
    __syncthreads();

    // ---- B: down projections, one warp per (circuit,row) pair ----
    const int warp = tid >> 5, lane = tid & 31;
    const int blocks = (d + 63) >> 6;
    for (int j = warp; j < totalRank; j += TA_THREADS / 32)
    {
        const unsigned long long* w = masks + (size_t)j * wprU64;
        int sum = 0;
        for (int b = lane; b < blocks; b += 32)
        {
            unsigned long long p = w[b * 2], m = w[b * 2 + 1];
            const signed char* xb = x8 + b * 64;
            while (p) { sum += xb[__ffsll((long long)p) - 1]; p &= p - 1; }
            while (m) { sum -= xb[__ffsll((long long)m) - 1]; m &= m - 1; }
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) rj[j] = fmaxf(0.0f, __fmul_rn((float)sum, ta_scale<SCALE_F16>(scales, j)));
    }
    __syncthreads();

    // ---- C: per circuit max / gate / requantize ----
    if (tid < nCirc)
    {
        const TaCircuitMeta cm = meta[tid];
        int open = 0;
        float dq = 0.0f;
        if (cm.alpha > 0.0f)
        {
            float m = 0.0f;
            for (int j = 0; j < cm.rank; j++) m = fmaxf(m, rj[cm.rankOffset + j]);
            if (m > fmaxf(cm.threshold, 1e-6f))
            {
                const float qsc = ta_div_rn(127.0f, m);
                for (int j = 0; j < cm.rank; j++)
                {
                    int q = (int)rintf(__fmul_rn(rj[cm.rankOffset + j], qsc));
                    qv[cm.rankOffset + j] = max(-128, min(127, q));
                }
                dq = __fmul_rn(ta_div_rn(m, 127.0f), cm.alpha);
                open = 1;
            }
        }
        openv[tid] = open;
        dqv[tid] = dq;
    }
    __syncthreads();
    int any = 0;
    for (int c = 0; c < nCirc; c++) any |= openv[c];
    if (!any) return;   // block-uniform: every circuit closed, row unchanged

    // ---- D: up projections + residual add; one thread owns 64 consecutive columns (one plus/minus word pair) ----
    const unsigned long long* upRows = masks + (size_t)totalRank * wprU64;
    const int colBase = tid * 64;
    if (colBase < d)
    {
        float tot[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) tot[i] = 0.0f;
        for (int c = 0; c < nCirc; c++)
        {
            if (!openv[c]) continue;
            const TaCircuitMeta cm = meta[c];
            int acc[64];
            #pragma unroll
            for (int i = 0; i < 64; i++) acc[i] = 0;
            for (int j = 0; j < cm.rank; j++)
            {
                const unsigned long long* r = upRows + (size_t)(cm.rankOffset + j) * wprU64 + (size_t)tid * 2;
                const unsigned long long pw = r[0], mw = r[1];
                const int q = qv[cm.rankOffset + j];
                #pragma unroll
                for (int i = 0; i < 64; i++)
                    acc[i] += (int)((pw >> i) & 1ull) * q - (int)((mw >> i) & 1ull) * q;
            }
            const float dq = dqv[c];
            const int sbase = upScaleBase + c * upScaleStride + colBase;
            #pragma unroll
            for (int i = 0; i < 64; i++)
                if (colBase + i < d)
                {
                    const float u = __fmul_rn((float)acc[i], ta_scale<SCALE_F16>(scales, sbase + i));
                    tot[i] = __fadd_rn(tot[i], __fmul_rn(u, dq));
                }
        }
        #pragma unroll
        for (int i = 0; i < 64; i++)
            if (colBase + i < d) row[colBase + i] = __fadd_rn(row[colBase + i], tot[i]);
    }
}

extern "C" __global__ void __launch_bounds__(TA_THREADS) ternary_adapter_residual_f32(
    float* h, const unsigned long long* masks, const float* scales, const TaCircuitMeta* meta,
    const int d, const int nRows, const int nCirc, const int totalRank, const int wprU64,
    const int upScaleBase, const int upScaleStride)
{
    ternary_adapter_body<false>(h, masks, scales, meta, d, nRows, nCirc, totalRank, wprU64, upScaleBase, upScaleStride);
}

extern "C" __global__ void __launch_bounds__(TA_THREADS) ternary_adapter_residual_f32_f16scales(
    float* h, const unsigned long long* masks, const half* scales, const TaCircuitMeta* meta,
    const int d, const int nRows, const int nCirc, const int totalRank, const int wprU64,
    const int upScaleBase, const int upScaleStride)
{
    ternary_adapter_body<true>(h, masks, scales, meta, d, nRows, nCirc, totalRank, wprU64, upScaleBase, upScaleStride);
}
