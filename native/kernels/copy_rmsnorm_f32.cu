// Fused "copy input to a residual buffer, then RMSNorm it" — FP32 throughout.
//
// The Qwen3HybridDense/Qwen3MoeHybrid CUDA decode path calls this exact pair back-to-back
// twice per layer: `cuMemcpyDtoDAsync(residual, hidden)` immediately followed by
// `rmsnorm_f32(hidden, weight, normOut)` — saving the pre-norm value for the later residual
// add, then normalizing the SAME input. Profiling found these two launches (each individually
// tiny — hiddenSize=5120 at Bonsai-27B scale) together cost ~9% of decode time, consistent with
// per-launch WDDM dispatch overhead dominating at this size. Fusing into one kernel halves the
// launch count for this pattern: pass 1 (sum-of-squares) also stores the fused residual copy
// (an extra vectorized store next to a load already happening — free), pass 2 is unchanged.
//
// Numerics: byte-for-byte the same reduction/scale math as rmsnorm_f32.cu (float2 vectorized
// loads/stores, __shfl_xor_sync symmetric warp reduction, fmaf-folded 1/n) — only the fused
// residual store is new. `residual_out` and `output` must be distinct from each other and from
// `input`'s aliasing (this kernel does NOT support in-place norm — the two RunSingleLayerBody
// call sites it replaces always use 3 distinct buffers: input=HiddenState, residual_out=Residual,
// output=NormOutput).

extern "C" __global__ void __launch_bounds__(256) copy_rmsnorm_f32(
    const float* __restrict__ input,
    float*       __restrict__ residual_out,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x   = input + (size_t)row * n;
    float*       res = residual_out + (size_t)row * n;
    float*       y   = output + (size_t)row * n;
    const int tid = threadIdx.x;
    const int n2 = n >> 1;

    const float2* __restrict__ x2   = reinterpret_cast<const float2*>(x);
    const float2* __restrict__ w2   = reinterpret_cast<const float2*>(weight);
    float2*       __restrict__ res2 = reinterpret_cast<float2*>(res);
    float2*       __restrict__ y2   = reinterpret_cast<float2*>(y);

    // ── Pass 1: sum of squares via float2 loads, fused residual copy ──
    float sum_sq = 0.0f;
    for (int i = tid; i < n2; i += blockDim.x)
    {
        float2 v = x2[i];
        res2[i] = v;
        sum_sq = fmaf(v.x, v.x, sum_sq);
        sum_sq = fmaf(v.y, v.y, sum_sq);
    }
    if ((n & 1) && tid == 0)
    {
        float v = x[n - 1];
        res[n - 1] = v;
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

    // ── Pass 2: vectorized scale (identical to rmsnorm_f32) ──
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
