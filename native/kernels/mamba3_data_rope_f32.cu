// mamba3_data_rope_f32.cu — Mamba-3 canonical data-dependent RoPE (issue #346).
//
// Direct CUDA-C port of native/vulkan/shaders/mamba3_data_rope_f32.comp, itself
// validated against the CPU-authoritative DotLLM.Cpu.Kernels.Mamba3DataRoPE.ExecuteCanonical
// (src/DotLLM.Cpu/Kernels/Mamba3DataRoPE.cs). One CUDA block per head; sequential loop
// over t inside the block (barrier()->__syncthreads()); NO_FMA (see build_ptx.bat) to
// match the CPU reference's plain MathF operations bit-for-bit-modulo-fp-reduction-order.
//
// Layout: b, c are [T, nRank, nHead, dState] row-major, mutated in place.
// anglesRaw is [T, numRopeAngles] (shared across rank & head). dt is [T, nHead].
// cumPrev/cumOut are [nHead, numRopeAngles].
//
// mode: 0 = Pairwise (SISO — pairs (v[2k], v[2k+1]) over the first 2*numRopeAngles
//            channels; tail passes through unchanged).
//       1 = Halved (MIMO — pairs (v[k], v[k+dState/2]) for k in [0, numRopeAngles);
//            remaining lanes of each half pass through unchanged).

#define WG_SIZE 64
#define MAX_ROPE_ANGLES 256

extern "C" __global__ void __launch_bounds__(WG_SIZE) mamba3_data_rope_f32(
    float* __restrict__ b, float* __restrict__ c,
    const float* __restrict__ anglesRaw, const float* __restrict__ dt,
    const float* __restrict__ cumPrev, float* __restrict__ cumOut,
    const int seqLen, const int nRank, const int nHead, const int dState,
    const int numRopeAngles, const int mode, const int hasCumPrev, const int writeCumOut)
{
    __shared__ float sharedCum[MAX_ROPE_ANGLES];
    __shared__ float sharedCos[MAX_ROPE_ANGLES];
    __shared__ float sharedSin[MAX_ROPE_ANGLES];

    const int h = blockIdx.x;
    if (h >= nHead) return;
    const int tid = threadIdx.x;
    const int nra = numRopeAngles;
    const int halfDState = dState >> 1;
    const int bcHeadStride = dState;
    const int bcRankStride = nHead * dState;
    const int bcTokenStride = nRank * bcRankStride;
    const float PI_F = 3.14159265358979323846f;
    const float TWO_PI = 2.0f * PI_F;
    const float INV_TWO_PI = 1.0f / TWO_PI;

    for (int k = tid; k < nra; k += WG_SIZE)
        sharedCum[k] = hasCumPrev ? cumPrev[h * nra + k] : 0.0f;
    __syncthreads();

    for (int t = 0; t < seqLen; t++)
    {
        float dtHere = dt[t * nHead + h];

        for (int k = tid; k < nra; k += WG_SIZE)
        {
            float raw = anglesRaw[t * nra + k];
            float tanhPi = tanhf(raw) * PI_F;
            float v = sharedCum[k] + dtHere * tanhPi;
            float floored = floorf(v * INV_TWO_PI);
            v = v - TWO_PI * floored;
            sharedCum[k] = v;
            sharedCos[k] = cosf(v);
            sharedSin[k] = sinf(v);
        }
        __syncthreads();

        for (int i = tid; i < nra; i += WG_SIZE)
        {
            float co = sharedCos[i];
            float si = sharedSin[i];
            int tokenBase = t * bcTokenStride;
            for (int r = 0; r < nRank; r++)
            {
                int bcBase = tokenBase + r * bcRankStride + h * bcHeadStride;
                int i0, i1;
                if (mode == 0)
                {
                    i0 = bcBase + 2 * i; i1 = i0 + 1;
                }
                else
                {
                    i0 = bcBase + i; i1 = bcBase + halfDState + i;
                }
                float be = b[i0]; float bo = b[i1];
                float ce = c[i0]; float co2 = c[i1];
                b[i0] = co * be - si * bo;   b[i1] = si * be + co * bo;
                c[i0] = co * ce - si * co2;  c[i1] = si * ce + co * co2;
            }
        }
        __syncthreads();
    }

    if (writeCumOut)
    {
        for (int k = tid; k < nra; k += WG_SIZE)
            cumOut[h * nra + k] = sharedCum[k];
    }
}
