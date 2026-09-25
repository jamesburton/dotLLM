// mamba3_ssd_scan_mimo_f32.cu — Mamba-3 canonical MIMO SSD scan (issue #346).
// Direct CUDA-C translation of the CPU-authoritative
// DotLLM.Cpu.Kernels.Mamba3CanonicalSsd.ExecuteMimo (src/DotLLM.Cpu/Kernels/Mamba3CanonicalSsd.cs).
// One CUDA block per head, threads stride over headDim (p); each thread's
// state-update-then-readout for its own p is fused into one loop (see this
// task's plan note for why this preserves the CPU's ordering guarantee).
// NO_FMA (see build_ptx.bat) for CPU bit-parity.
//
// What is and is not bit-exact: like the SISO kernel
// (mamba3_ssd_scan_siso_f32.cu), operation order and non-fusion (no FMA on the
// caller's own mul/add pairs — NO_FMA above) are preserved exactly versus the
// CPU reference (Mamba3CanonicalSsd.ExecuteMimo). That is NOT the same as a
// whole-kernel "CPU-bit-parity" claim: decay = expf(adt[...]) feeds
// `newState = decay * state + vp * (kSum * scl)` at every token, and the
// optional gate's expf(-zGated) is a second transcendental site — CUDA's
// precise libdevice expf is not guaranteed bit-identical to .NET's
// MathF.Exp for the same input (IEEE 754 does not mandate correctly-rounded
// transcendentals). Because this kernel folds the ENTIRE sequence into one
// launch with a sequential per-token state update, any sub-ULP drift
// introduced at t=0 compounds through every later token's state and y —
// same class/scale of drift as the SISO kernel. Expect O(ulp)-scale drift on
// state/y outputs; the unit test uses a documented tolerance, not bit-exact
// SequenceEqual.
//
// state:  [nHead, headDim, dState] F32, mutated in place, rank-free (K is
//         summed over rank INSIDE the state update — matches ExecuteMimo's
//         "h is shape [nHead,headDim,dState], no rank dim" contract).
// v:      [seqLen, nHead, headDim] F32.
// qRoped/kRoped: [seqLen, nRank, nHead, dState] F32 (post-RoPE C / B per rank).
// qkPreDotSum, scale, gamma, adt: [seqLen, nHead] F32 (host-precomputed).
// d: [nHead] F32. z: [seqLen, nHead, headDim] F32 or nullptr. mimoZ/mimoO: [nHead, nRank, headDim] F32.
// y: [seqLen, nHead, headDim] F32 output (rank-contracted).

#define SSD_MIMO_WG_SIZE 256

extern "C" __global__ void __launch_bounds__(SSD_MIMO_WG_SIZE) mamba3_ssd_scan_mimo_f32(
    float* __restrict__ state,
    const float* __restrict__ v,
    const float* __restrict__ qRoped,
    const float* __restrict__ kRoped,
    const float* __restrict__ qkPreDotSum,
    const float* __restrict__ scale,
    const float* __restrict__ gamma,
    const float* __restrict__ adt,
    const float* __restrict__ d,
    const float* __restrict__ z,
    const float* __restrict__ mimoZ,
    const float* __restrict__ mimoO,
    float* __restrict__ y,
    const int seqLen, const int nRank, const int nHead, const int headDim, const int dState, const int hasZ)
{
    __shared__ float decayShared, sclShared, gmShared, qkpShared, skipShared;

    const int h = blockIdx.x;
    if (h >= nHead) return;
    const int tid = threadIdx.x;
    const int stateHeadBase = h * headDim * dState;
    const float dh = d[h];
    const float invRank = 1.0f / (float)nRank;

    const int bcHeadStride = dState;
    const int bcRankStride = nHead * dState;
    const int bcTokStride = nRank * bcRankStride;
    const int mimoHeadStride = nRank * headDim;
    const int mimoRankStride = headDim;

    for (int t = 0; t < seqLen; t++)
    {
        if (tid == 0)
        {
            int hdrIdx = t * nHead + h;
            decayShared = expf(adt[hdrIdx]);
            sclShared = scale[hdrIdx];
            gmShared = gamma[hdrIdx];
            qkpShared = qkPreDotSum[hdrIdx];
            skipShared = dh + gmShared * qkpShared;
        }
        __syncthreads();

        float decay = decayShared, scl = sclShared, skip = skipShared;
        int vTokBase = t * nHead * headDim;
        int bcTokBase = t * bcTokStride;

        for (int p = tid; p < headDim; p += SSD_MIMO_WG_SIZE)
        {
            int vIdx = vTokBase + h * headDim + p;
            float vp = v[vIdx];
            int stateRowBase = stateHeadBase + p * dState;

            // State update: h_new[p,n] = decay*h_old[p,n] + vp*(sum_r kRoped[t,r,h,n])*scl.
            for (int n = 0; n < dState; n++)
            {
                float kSum = 0.0f;
                for (int r = 0; r < nRank; r++)
                {
                    int kIdx = bcTokBase + r * bcRankStride + h * bcHeadStride + n;
                    kSum += kRoped[kIdx];
                }
                float newState = decay * state[stateRowBase + n] + vp * (kSum * scl);
                state[stateRowBase + n] = newState;
            }

            // Per-rank readout + gate + rank contraction (same thread, same p — the
            // state row this thread just finished writing is the one it reads here).
            float contracted = 0.0f;
            for (int r = 0; r < nRank; r++)
            {
                int qBase = bcTokBase + r * bcRankStride + h * bcHeadStride;
                float yScanR = 0.0f;
                for (int n = 0; n < dState; n++)
                    yScanR += qRoped[qBase + n] * state[stateRowBase + n];

                float yR = yScanR + skip * invRank * vp;

                if (hasZ)
                {
                    int mimoZIdx = h * mimoHeadStride + r * mimoRankStride + p;
                    float zGated = z[vIdx] * mimoZ[mimoZIdx];
                    float silu = zGated / (1.0f + expf(-zGated));
                    yR *= silu;
                }

                int mimoOIdx = h * mimoHeadStride + r * mimoRankStride + p;
                contracted += yR * mimoO[mimoOIdx];
            }
            y[vIdx] = contracted;
        }
        __syncthreads();
    }
}
