// PrismML blockwise normalized Walsh-Hadamard activation transform (prism.hadamard.*).
//
// Bonsai 2 stores its ternary weights in a rotated basis, so every folded weight's input
// activation must be rotated to match before the matmul. This kernel is that rotation. It is the
// CUDA twin of DotLLM.Cpu.Kernels.Hadamard (the parity oracle) and of the Vulkan shader
// native/vulkan/shaders/hadamard_fwht_f32.comp, which it mirrors line for line.
//
// Operator: a Sylvester-ordered Walsh-Hadamard transform scaled by 1/sqrt(blockSize), applied
// independently to each contiguous blockSize-wide block of a row. Computed as an in-place
// butterfly in shared memory, so the rotation matrix is never materialized and no weight bytes
// move. Cost is O(n log n) over the activation alone.
//
// Three composable steps, in an order that DIFFERS between the two directions:
//
//   forward (a folded weight's input):   [optional GDN permute] -> signs -> FWHT
//   inverse (a rotated lookup row):      FWHT -> signs
//
// The rotation is its own inverse, so (H.S)^-1 = S.H. Applying the signs on the wrong side is
// silent corruption, not a crash.
//
// The GDN permute (prism.hadamard.gdn_v_grouped, *.ssm_out.weight only) reorders value heads from
// the recurrence's tiled order vh = k + nKHead*r into the grouped order vh' = r + rep*k that the
// fold was computed in. It is folded into the LOAD as an index remap and reaches across block
// boundaries within the row, so src must NOT alias dst when permute != 0. Signs are indexed by the
// destination (post-permute) column, matching Hadamard.PermuteTiledToGrouped + ForwardRow.
//
// Without the permute each CUDA block reads only the block it writes, and every read completes
// before the first __syncthreads(), so src == dst (in-place) is safe.
//
// Numerics: the 1/sqrt(n) scale is passed in from the host (1f / MathF.Sqrt(n), the exact value
// the CPU oracle uses) rather than computed with rsqrtf, and the butterflies are pure add/sub, so
// the result is bit-identical to the CPU scalar/SIMD butterfly when compiled with -fmad=false
// (listed as such in native/build.ps1, native/build_ptx.bat and DotLLM.Cuda.csproj).
//
// Launch:
//   grid  = (width / blockSize, rows, 1)
//   block = (HADAMARD_THREADS, 1, 1)
//   blockSize must be a power of two, 2 <= blockSize <= HADAMARD_MAX_BLOCK (checked on the host).

#define HADAMARD_MAX_BLOCK 1024
#define HADAMARD_THREADS 256

extern "C" __global__ void hadamard_fwht_f32(
    const float* src,                  // may alias dst when permute == 0 (hence no __restrict__)
    float* dst,
    const float* __restrict__ signs,   // [width], +/-1; ignored when applySigns == 0
    int rows,
    int width,          // full row width (multiple of blockSize)
    int blockSize,      // Hadamard block width, power of two
    int applySigns,     // 0 = identity sign step
    int inverse,        // 0 = signs before FWHT (forward), 1 = signs after (inverse)
    int permute,        // 1 = GDN tiled -> grouped value-head remap on load
    int permDState,     // head width for the permute
    int permNKHead,     // key heads
    int permRep,        // value heads per key head
    float scale)        // 1/sqrt(blockSize), computed on the host
{
    __shared__ float s[HADAMARD_MAX_BLOCK];

    const int blk = blockIdx.x;
    const int row = blockIdx.y;
    const int blocksPerRow = width / blockSize;
    if (row >= rows || blk >= blocksPerRow) return;   // uniform per CUDA block — no barrier hazard

    const int n = blockSize;
    const long long rowBase = (long long)row * width;
    const int blockOff = blk * n;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    // Load: optional permute, forward-direction signs, and the 1/sqrt(n) scale — one multiply
    // per element on the way into shared memory, in the same order as the CPU oracle
    // (src * sign) * scale == (+/-src) * scale.
    for (int i = tid; i < n; i += threads)
    {
        const int col = blockOff + i;   // column within the row, post-permute

        int srcCol = col;
        if (permute != 0)
        {
            // dest head dh reads source head (dh / rep) + nKHead * (dh % rep)
            const int dh = col / permDState;
            const int lane = col - dh * permDState;
            const int srcHead = (dh / permRep) + permNKHead * (dh % permRep);
            srcCol = srcHead * permDState + lane;
        }

        float v = src[rowBase + srcCol];
        if (applySigns != 0 && inverse == 0 && signs[col] < 0.0f)
            v = -v;

        s[i] = v * scale;
    }
    __syncthreads();

    // Sylvester-ordered butterfly passes. Every pass is a full shared-memory sweep, so each needs
    // a barrier after it: the reads of pass k+1 must not race the writes of pass k.
    for (int len = 1; len < n; len <<= 1)
    {
        for (int b = tid; b < (n >> 1); b += threads)
        {
            const int span = b / len;              // which 2*len-wide span
            const int j = b - span * len;
            const int i0 = span * (len << 1) + j;
            const int i1 = i0 + len;

            const float u = s[i0];
            const float w = s[i1];
            s[i0] = u + w;
            s[i1] = u - w;
        }
        __syncthreads();
    }

    for (int i = tid; i < n; i += threads)
    {
        const int col = blockOff + i;
        float v = s[i];
        if (applySigns != 0 && inverse != 0 && signs[col] < 0.0f)
            v = -v;
        dst[rowBase + col] = v;
    }
}

// ─── blockSize == 1024 fast path (issue #1222) ──────────────────────────────────────────────────
// Same transform, same butterflies, fewer barriers. The generic kernel above runs all 10 passes
// through shared memory with a barrier (and a runtime integer divide) per pass. Here each of the 256
// threads owns 4 consecutive elements: passes len = 1, 2 stay in registers, len = 4..64 pair threads
// 1..16 lanes apart inside one warp (__shfl_xor), and only len = 128, 256, 512 cross warps — 128
// threads then each take the 8 elements t + 128*j and finish those passes in registers.
//
// Every butterfly is still s[i0] + s[i1] / s[i0] - s[i1] on the same pair in the same pass order,
// so the output is bit-identical to hadamard_fwht_f32 (and the CPU oracle): only where each operand
// sits changes. The lower element of a pair is the one whose index has the pass bit clear.
#define HADAMARD1024_N 1024

extern "C" __global__ void __launch_bounds__(HADAMARD_THREADS) hadamard_fwht1024_f32(
    const float* src,
    float* dst,
    const float* __restrict__ signs,
    int rows,
    int width,
    int blockSize,      // == 1024 (host-checked)
    int applySigns,
    int inverse,
    int permute,
    int permDState,
    int permNKHead,
    int permRep,
    float scale)
{
    __shared__ float s[HADAMARD1024_N];

    const int blk = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || blk >= width / HADAMARD1024_N) return;

    const long long rowBase = (long long)row * width;
    const int blockOff = blk * HADAMARD1024_N;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    float v[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
    {
        const int col = blockOff + 4 * tid + k;
        int srcCol = col;
        if (permute != 0)
        {
            const int dh = col / permDState;
            const int ln = col - dh * permDState;
            srcCol = ((dh / permRep) + permNKHead * (dh % permRep)) * permDState + ln;
        }
        float x = src[rowBase + srcCol];
        if (applySigns != 0 && inverse == 0 && signs[col] < 0.0f)
            x = -x;
        v[k] = x * scale;
    }

    // len = 1, 2: within the thread's 4 elements.
    {
        float a = v[0], b = v[1], c = v[2], d = v[3];
        v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
        a = v[0]; b = v[1]; c = v[2]; d = v[3];
        v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
    }

    // len = 4 .. 64: the partner is lane ^ m, m = len / 4.
    #pragma unroll
    for (int m = 1; m <= 16; m <<= 1)
    {
        const bool upper = (lane & m) != 0;
        #pragma unroll
        for (int k = 0; k < 4; k++)
        {
            const float p = __shfl_xor_sync(0xFFFFFFFFu, v[k], m);
            v[k] = upper ? p - v[k] : v[k] + p;
        }
    }

    #pragma unroll
    for (int k = 0; k < 4; k++)
        s[4 * tid + k] = v[k];
    __syncthreads();

    if (tid < 128)
    {
        // len = 128, 256, 512: elements tid + 128*j, j = 0..7; the pass bit is j's bit.
        float u[8];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            u[j] = s[tid + 128 * j];
        #pragma unroll
        for (int bit = 1; bit < 8; bit <<= 1)
        {
            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                if ((j & bit) == 0)
                {
                    const float a = u[j], b = u[j | bit];
                    u[j] = a + b;
                    u[j | bit] = a - b;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const int col = blockOff + tid + 128 * j;
            float x = u[j];
            if (applySigns != 0 && inverse != 0 && signs[col] < 0.0f)
                x = -x;
            dst[rowBase + col] = x;
        }
    }
}
