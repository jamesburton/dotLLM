// mamba3_chunk_boundary_f32.cu — Mamba-3 streaming-decode chunk-boundary state
// correction (issue #346). Direct CUDA-C port of
// native/vulkan/shaders/mamba3_chunk_boundary_f32.comp. Fully parallel over
// (head, headDim, dState) — no time recurrence, unlike the SSD scan kernels.
// NO_FMA (see build_ptx.bat) for CPU-bit-parity with
// Mamba3Block.ApplyChunkBoundaryAdjustment / Mamba3CanonicalSsd.ExecuteMimoStreaming's
// boundary block.
//
// state:  [nHead, headDim, dState] F32, mutated in place (+=).
// vState: [nHead, headDim] F32, previous chunk's last-token V.
// kState: [nRank, nHead, dState] F32, previous chunk's last-token post-RoPE K
//         (nRank=1 for SISO).
// coef:   [nHead] F32, precomputed dt[0,h]*(1-trap[0,h]) — computed host-side
//         during the per-token preprocessing step (Task 9), NOT inside this kernel.
//
// ssm_state[h,p,n] += vState[h,p] * (sum_r kState[r,h,n]) * coef[h]

#define BOUNDARY_WG_X 16
#define BOUNDARY_WG_Y 16

extern "C" __global__ void __launch_bounds__(BOUNDARY_WG_X * BOUNDARY_WG_Y) mamba3_chunk_boundary_f32(
    float* __restrict__ state,
    const float* __restrict__ vState,
    const float* __restrict__ kState,
    const float* __restrict__ coef,
    const int nHead, const int headDim, const int dState, const int nRank)
{
    const int n = blockIdx.x * BOUNDARY_WG_X + threadIdx.x;
    const int p = blockIdx.y * BOUNDARY_WG_Y + threadIdx.y;
    const int h = blockIdx.z;
    if (h >= nHead || p >= headDim || n >= dState) return;

    float c = coef[h];
    if (c == 0.0f) return;

    int kRankStride = nHead * dState;
    int kHeadOff = h * dState + n;
    float kSum = 0.0f;
    for (int r = 0; r < nRank; r++)
        kSum += kState[r * kRankStride + kHeadOff];

    float v = vState[h * headDim + p];
    int stateIdx = h * headDim * dState + p * dState + n;
    state[stateIdx] += v * kSum * c;
}
