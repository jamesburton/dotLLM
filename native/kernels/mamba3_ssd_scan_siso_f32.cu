// mamba3_ssd_scan_siso_f32.cu — Mamba-3 canonical SISO SSD scan (issue #346).
// Direct CUDA-C port of native/vulkan/shaders/mamba3_canonical_ssd_siso_f32.comp,
// itself validated against DotLLM.Cpu.Kernels.Mamba3CanonicalSsd.ExecuteSiso
// (src/DotLLM.Cpu/Kernels/Mamba3CanonicalSsd.cs). One CUDA block per head; each
// thread strides over headDim (p) and owns the full dState-wide (p, :) state row
// across the entire sequential t loop, iterating n serially inside its own thread —
// this reproduces the CPU reference's `for(p){for(n){...}}` nesting exactly, so no
// cross-thread reduction exists in the recurrence itself and per-thread mul/add
// accumulation order (state update, yScan reduction, the kScaled/skip products)
// matches the CPU scalar reference term-for-term. NO_FMA (see build_ptx.bat) so
// nvcc does not fuse any of those mul/add pairs into a single FMA the CPU's plain
// MathF arithmetic does not perform.
//
// What is and is not bit-exact: operation order and non-fusion (no FMA on the
// caller's own mul/add pairs — see NO_FMA above) are preserved exactly versus
// the CPU reference. That is NOT the same as a whole-kernel "CPU-bit-parity"
// claim: decay = expf(adt[...]) feeds `s = decay * state + vp * kScaled` at
// every token, so any libdevice-vs-CPU-MathF.Exp transcendental rounding
// difference enters the state recurrence at t=0 and compounds across the
// sequential t loop — nothing downstream of decay (state, yScan, y) is
// guaranteed bit-exact as a result, only order-preserving. Task 1/2's
// (mamba3_data_rope_f32) real-GPU testing found ~1-ULP libdevice-vs-MathF
// drift on expf/cosf/sinf even with -fmad=false, because that drift lives
// inside libdevice's own polynomial/reduction implementation, not in FMA
// fusion of the caller's arithmetic. The optional silu(z) gate's expf(-zv) is
// a second, independent transcendental source but does not feed back into
// state. Expect O(ulp)-scale drift on state/y outputs; Task 6's tests need a
// documented tolerance (as Task 2's did), not bit-exact SequenceEqual.
//
// state:  [nHead, headDim, dState] F32, mutated in place across the whole call
//         (this is a single kernel launch covering ALL seqLen tokens, unlike the
//         GDN scan kernel's one-launch-per-token host loop).
// v:      [seqLen, nHead, headDim] F32 (the SSM "x" / value input).
// qRoped/kRoped: [seqLen, nHead, dState] F32 (post-RoPE C / B).
// qkPreDot, scale, gamma, adt: [seqLen, nHead] F32 (host-precomputed, see Task 9).
// d:      [nHead] F32 skip coefficient.
// z:      [seqLen, nHead, headDim] F32 gate input, or nullptr when hasZ==0.
// y:      [seqLen, nHead, headDim] F32 output.

#define SSD_WG_SIZE 256

extern "C" __global__ void __launch_bounds__(SSD_WG_SIZE) mamba3_ssd_scan_siso_f32(
    float* __restrict__ state,
    const float* __restrict__ v,
    const float* __restrict__ qRoped,
    const float* __restrict__ kRoped,
    const float* __restrict__ qkPreDot,
    const float* __restrict__ scale,
    const float* __restrict__ gamma,
    const float* __restrict__ adt,
    const float* __restrict__ d,
    const float* __restrict__ z,
    float* __restrict__ y,
    const int seqLen, const int nHead, const int headDim, const int dState, const int hasZ)
{
    __shared__ float decayShared, sclShared, skipShared;

    const int h = blockIdx.x;
    if (h >= nHead) return;
    const int tid = threadIdx.x;
    const int stateHeadBase = h * headDim * dState;
    const float dh = d[h];

    for (int t = 0; t < seqLen; t++)
    {
        if (tid == 0)
        {
            int hdrIdx = t * nHead + h;
            float decay = expf(adt[hdrIdx]);
            float scl_ = scale[hdrIdx];
            float gm_ = gamma[hdrIdx];
            float qkp_ = qkPreDot[hdrIdx];
            decayShared = decay;
            sclShared = scl_;
            skipShared = dh + gm_ * qkp_;
        }
        __syncthreads();

        float decay = decayShared;
        float scaleT = sclShared;
        float skipT = skipShared;

        int vRowBase = t * nHead * headDim + h * headDim;
        int bcRowBase = t * nHead * dState + h * dState;
        int yRowBase = vRowBase;
        int zRowBase = vRowBase;

        for (int p = tid; p < headDim; p += SSD_WG_SIZE)
        {
            float vp = v[vRowBase + p];
            int stateRowBase = stateHeadBase + p * dState;
            float yScan = 0.0f;
            for (int n = 0; n < dState; n++)
            {
                float kScaled = kRoped[bcRowBase + n] * scaleT;
                float s = decay * state[stateRowBase + n] + vp * kScaled;
                state[stateRowBase + n] = s;
                yScan += qRoped[bcRowBase + n] * s;
            }

            float yOut = yScan + skipT * vp;
            if (hasZ)
            {
                float zv = z[zRowBase + p];
                float siluZ = zv / (1.0f + expf(-zv));
                yOut *= siluZ;
            }
            y[yRowBase + p] = yOut;
        }
        __syncthreads();
    }
}
