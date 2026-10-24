// Pointwise FP32 element-wise kernels used by the Qwen3MoeHybrid recurrence /
// full-attention paths. Each is a bit-perfect port of the corresponding host
// fallback in CudaQwen3MoeHybridTransformerModel.cs (LaunchSigmoidHostFallback,
// LaunchSiluHostFallback, LaunchSigmoidMulHostFallback).
//
// All three call expf — compiled with -fmad=false (see build_ptx.bat NO_FMA
// list) so the multiply/add patterns around the sigmoid don't get fused. CUDA's
// precise expf is within ≤1 ULP of MathF.Exp on Ampere+, so the output is
// numerically equivalent to the CPU host-side reference; not strictly
// bit-equal across all inputs, but the largest divergence observed on uniform
// [-4, 4] inputs is ≤ 2 × FLT_EPSILON. Tests admit a small ULP tolerance.
//
// Grid-stride loop pattern: each kernel launches with enough blocks to cover
// total / blockDim — but uses `idx < total` rather than a stride loop. For
// typical Qwen3MoeHybrid sizes (seq_len × n_v_head ≤ a few thousand) the
// grid is small and the simple form is faster.

#include <math.h>

extern "C" __global__ void sigmoid_f32(
    float* __restrict__ buf,                // in/out, [n]
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // 1 / (1 + exp(-x)) — same form as the CPU host fallback. Avoid the
    // 0.5f * (1f + tanh(0.5*x)) form: it would diverge from MathF.Exp.
    buf[idx] = 1.0f / (1.0f + expf(-buf[idx]));
}

extern "C" __global__ void silu_f32(
    float* __restrict__ buf,                // in/out, [n]
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // silu(x) = x * sigmoid(x). Match the host fallback ordering exactly:
    // x * (1f / (1f + exp(-x))) — NOT x / (1 + exp(-x)) (different rounding).
    float x = buf[idx];
    buf[idx] = x * (1.0f / (1.0f + expf(-x)));
}

extern "C" __global__ void sigmoid_mul_f32(
    float* __restrict__ a,                   // in/out, [n]
    const float* __restrict__ b,             // [n]
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // out_i *= sigmoid(b_i). Match the host fallback: a[i] *= 1f/(1f+exp(-b[i])).
    float bi = b[idx];
    a[idx] = a[idx] * (1.0f / (1.0f + expf(-bi)));
}

// ── De-interleave gather kernels ──────────────────────────────────────────
// Replace decode-time host loops that issued numHeads (or seqLen) separate
// cuMemcpyDtoDAsync calls per layer to split a fused projection's output
// into its logical sub-tensors. On Windows/WDDM each async memcpy still
// costs real host-side driver-call overhead even though it's "async" — for
// Bonsai-27B's ~40 attention heads that was 80 launches per attention layer
// (16 layers/model), measured as the single biggest non-GEMV decode cost
// (profiled ~12% of total decode time, more than the actual attention
// kernel itself). One gather kernel launch replaces the whole per-head loop.

// Q+Gate de-interleave (full-attention layers). Per-token qg layout:
// [Q_h0(headDim), Gate_h0(headDim), Q_h1(headDim), Gate_h1(headDim), ...] —
// each head is 2*headDim contiguous floats, Q first then Gate.
extern "C" __global__ void deinterleave_qgate_f32(
    const float* __restrict__ qg,     // [seqLen, 2*numHeads*headDim]
    float* __restrict__ q,            // [seqLen, numHeads*headDim]
    float* __restrict__ gate,         // [seqLen, numHeads*headDim]
    const int numHeads,
    const int headDim,
    const int seqLen)
{
    int qElems = numHeads * headDim;
    long total = (long)seqLen * qElems;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int t = (int)(idx / qElems);
    int e = (int)(idx % qElems);
    int h = e / headDim;
    int d = e % headDim;
    long qgBase = (long)t * 2 * qElems + (long)h * 2 * headDim;
    q[idx] = qg[qgBase + d];
    gate[idx] = qg[qgBase + headDim + d];
}

// GDN Q/K/V de-interleave. Per-token src layout: [Q(kDim) | K(kDim) | V(vDim)].
extern "C" __global__ void deinterleave_gdn_qkv_f32(
    const float* __restrict__ src,    // [seqLen, 2*kDim+vDim]
    float* __restrict__ q,            // [seqLen, kDim]
    float* __restrict__ k,            // [seqLen, kDim]
    float* __restrict__ v,            // [seqLen, vDim]
    const int kDim,
    const int vDim,
    const int seqLen)
{
    int convDim = 2 * kDim + vDim;
    long total = (long)seqLen * convDim;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int t = (int)(idx / convDim);
    int e = (int)(idx % convDim);
    float val = src[idx];
    if (e < kDim)
        q[(long)t * kDim + e] = val;
    else if (e < 2 * kDim)
        k[(long)t * kDim + (e - kDim)] = val;
    else
        v[(long)t * vDim + (e - 2 * kDim)] = val;
}
