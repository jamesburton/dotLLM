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
