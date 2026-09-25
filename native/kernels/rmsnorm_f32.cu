// Full FP32 RMS Normalization: FP32 input, FP32 weight, FP32 output.
//
// Optimizations:
//   * float2 vectorized loads/stores
//   * __shfl_xor_sync warp reduction (symmetric)
//   * Pre-folds 1/n into the rsqrt argument via fmaf
//
// ─── Investigated (#172): fusing this + swiglu_f32 for the GDN normgate call site — NEGATIVE ───
// The Qwen3HybridDense/Qwen3MoeHybrid GDN body (ForwardGdnBody step 6, "gdn-6-normgate" in the
// DOTLLM_HYBRID_PROFILE category profiler) calls this kernel immediately followed by swiglu_f32
// (see swiglu_f32.cu for the matching note) — a per-head RMSNorm(x, ssm_norm) then silu(z)-gate,
// ~9.56ms per 4 decode steps in the post-#170 profile (48 GDN layers x 4 steps). Implemented and
// SASS-verified a fused rmsnorm_swiglu_f32 kernel (one block per (token, v-head) row, same
// float2/shfl_xor reduction as this kernel, gate fused into pass 2 — structurally the same idea as
// copy_rmsnorm_f32.cu's residual-copy fusion, just fusing the *next* op instead of the *previous*
// one). ptxas -v showed a clean compile (32 registers, 0 spill, vs. 24 for this kernel alone — no
// occupancy concern). A bit-exact-on-the-norm/~3e-7-relative-on-the-gate correctness test passed
// for all tested shapes (decode-realistic nVHead=48/dState=128, prefill-shaped seqLen>1, small
// dims), confirming the fusion was implemented correctly.
//
// MEASURED REAL BONSAI-27B DECODE THROUGHPUT SHOWED NO REPRODUCIBLE IMPROVEMENT: five `bench
// -p 64 -n 48` runs (2 baseline, 3 fused, RTX 3060) all landed in the same ~17.5-18.3 tok/s band
// regardless of which kernel path was active — differences between fused and baseline were smaller
// than the run-to-run spread (itself dominated by within-run thermal drift: decode time on this
// card visibly rises rep-over-rep within a single `bench` invocation as the GPU heats up, e.g.
// 2646ms -> 2885ms across 12 reps in one run). "Best" (least-throttled) reps clustered at
// 17.97-18.25 tok/s for BOTH baseline and fused with no consistent ordering. Consistent with this
// investigation's own prediction going in: this bucket's absolute time share (~9.56ms/4 steps,
// smaller than #170's ~16ms target which itself only moved the needle ~0.7-1%) was expected to
// yield a small result if any — in practice it was too small to separate from this machine's
// thermal-noise floor. NOT reverted due to a regression (no regression was observed either) — this
// is a "correctly implemented, real-world benefit unmeasurable" negative result. Reverted in full
// (no rmsnorm_swiglu_f32 kernel/wiring/test shipped) to avoid carrying unused code for a change
// that cannot be shown to help. Don't re-attempt this exact fusion without either a lower-noise
// benching setup (e.g. a dedicated cool-GPU environment) or a larger-absolute-time-share motivation
// (e.g. if a future model config makes nVHead or dState much larger).

extern "C" __global__ void __launch_bounds__(256) rmsnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    float* y = output + (size_t)row * n;
    const int tid = threadIdx.x;
    const int n2 = n >> 1;

    const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
    const float2* __restrict__ w2 = reinterpret_cast<const float2*>(weight);
    float2* __restrict__       y2 = reinterpret_cast<float2*>(y);

    // ── Pass 1: sum of squares via float2 loads ──
    float sum_sq = 0.0f;
    for (int i = tid; i < n2; i += blockDim.x)
    {
        float2 v = x2[i];
        sum_sq = fmaf(v.x, v.x, sum_sq);
        sum_sq = fmaf(v.y, v.y, sum_sq);
    }
    if ((n & 1) && tid == 0)
    {
        float v = x[n - 1];
        sum_sq = fmaf(v, v, sum_sq);
    }

    // ── Warp reduction (symmetric) ──
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    __shared__ float ws[32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) ws[wid] = sum_sq;
    __syncthreads();

    if (wid == 0)
    {
        int nw = (blockDim.x + 31) >> 5;
        sum_sq = (lane < nw) ? ws[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
        if (lane == 0)
            ws[0] = rsqrtf(fmaf(sum_sq, 1.0f / (float)n, eps));
    }
    __syncthreads();
    const float ri = ws[0];

    // ── Pass 2: vectorized scale ──
    for (int i = tid; i < n2; i += blockDim.x)
    {
        float2 v = x2[i];
        float2 wh = w2[i];
        float2 r;
        r.x = v.x * ri * wh.x;
        r.y = v.y * ri * wh.y;
        y2[i] = r;
    }
    if ((n & 1) && tid == 0)
    {
        int last = n - 1;
        y[last] = x[last] * ri * weight[last];
    }
}
