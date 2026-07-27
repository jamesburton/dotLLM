// Fused SwiGLU with FP32 data.
//
// Investigated (#172): fusing this kernel with the preceding rmsnorm_f32 call at the GDN normgate
// call site (ForwardGdnBody step 6) — implemented, SASS-verified, correctness-tested, but showed
// NO reproducible real-bench decode throughput improvement on Bonsai-27B (differences smaller than
// this machine's run-to-run thermal-drift noise). Reverted in full. See rmsnorm_f32.cu's header
// for the full writeup (measured numbers, root-cause reasoning, and why nothing shipped here).
// Note this kernel compiles with --use_fast_math (see native/build_ptx.bat's FAST_MATH list) — its
// SiLU sigmoid lowers to MUFU.EX2 (approximate hardware exp2) + MUFU.RCP, not precise expf(); any
// future fusion attempt touching this kernel's math must account for that precision difference
// (measured ~2-4e-7 relative error vs. precise expf() in the #172 correctness test, not ULP-scale).

#include <math.h>

extern "C" __global__ void __launch_bounds__(256) swiglu_f32(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, const int n, const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * seq_len)
    {
        float g = gate[idx], u = up[idx];
        output[idx] = (g / (1.0f + expf(-g))) * u;
    }
}
