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
