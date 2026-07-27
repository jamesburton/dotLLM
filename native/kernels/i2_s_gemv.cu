// I2_S (BitNet b1.58 ternary) GEMV for dotLLM's decode path.
// y[row] = scale * sum_k (code(W[row,k]) - 1) * x[k]
//
// dotLLM I2_S on-disk layout (must match MatMul.I2S.cs / Dequantize.DequantizeI2_S):
//   * Row-major W[n,k], k a multiple of 128. Row stride = k/4 bytes (4 codes/byte, 2 bits each).
//   * 128-element block = 32 bytes. Byte gp in [0,31] holds elements {gp, gp+32, gp+64, gp+96}
//     at bit offsets {6,4,2,0}.
//   * Code mapping value = code - 1  (0->-1, 1->0, 2->+1). NOTE: differs from BitNet GPU's
//     offset-binary {1,2,3}; the decode here subtracts 1.
//   * ONE per-tensor float32 scale at the tensor tail, byte offset (size_t)n*(k/4).
//
// ───────────────────────── Occupancy / MLP optimization (v2: warp-per-row) ─────────────────────────
// HISTORY. v1 used grid=(n,1,1)/block=(256,1,1) with ONE block per output row, each thread striding the
// row's uint4 units. But a row is only k/4 bytes = k/64 uint4 units = 40 (k=2560) .. 108 (k=6912) units,
// so only 40..108 of the block's 256 threads ever issued a load — 6 of the 8 warps sat idle, the block
// reduce summed mostly-empty warps, and we launched n=2560 such half-empty blocks. Measured 22 GB/s
// (k=2560) .. 29 GB/s (k=6912), ~6-8% of the 3060's 360 GB/s peak.
//
// v2 packs MULTIPLE rows per block (à la Microsoft BitNet's 16-rows/block GPU kernel) so every warp does
// real work and the shared-loaded x is reused across all rows in the block:
//   * block = 256 threads = 8 warps; ROWS_PER_BLOCK = 8 → ONE WARP OWNS ONE OUTPUT ROW.
//   * grid.x = ceil(n / 8). Block b covers rows [8b, 8b+8). Warp wid (=threadIdx.x/32) owns row 8b+wid.
//   * x[k] is staged into shared ONCE by all 256 threads, then read by all 8 rows' warps → the shared
//     stage cost (and the half→float x traffic) is amortized 8× vs one-row-per-block.
//   * Each warp's 32 lanes stride over the row's 40..108 uint4 units (1..4 units/lane) → the warp is
//     fully populated and issues 32 in-flight 16-byte loads (ld.global.nc.v4.u32) at a time. Across the
//     8 warps that is up to 256 independent weight loads in flight per block — far higher memory-level
//     parallelism than v1, and the weight matrix (the bandwidth-bound operand) is still read exactly once.
//   * Reduction is a single intra-warp __shfl_down (no __syncthreads, no shared warp_sums, no idle-warp
//     summing) — lane 0 of each warp writes its row.
//
// Each warp reads weights as uint4 (16 B = 64 codes) per load — wide, fully-coalesced transactions. For
// FP16/FP32 activations x is shared-resident; the int8/dp4a variant reads xq from global (__ldg,
// L2-resident) because its {gp,+32,+64,+96} single-byte gathers serialize through shared (measured
// slower) while the per-token xq stays L2-hot across all blocks anyway.
//
// Numerics are unchanged (exact ternary decode, fp32/int32 accumulate). Only the reduction is now a pure
// 32-lane warp reduce; the fp32 sum order stays within the test's <=1e-3 tolerance (measured max abs diff
// ~1e-6 vs the CPU float reference).
//
// Layout note for the uint4 load: row_bytes = k/4 is a multiple of 32 (k%128==0), hence a multiple of 16,
// so it splits cleanly into uint4 (16-byte) units. A uint4 spans bytes [16u, 16u+15]; since 16u is a
// multiple of 16, those 16 bytes lie inside a single 32-byte (128-element) block — blk and the x base
// address are constant across the uint4, only gp = byte index within the block varies.
//
// Launch contract (set in CudaKernels.cs): block = (256,1,1); grid = (ceil(n/ROWS_PER_BLOCK),1,1);
// shared = k * sizeof(float) bytes (dynamic — see below).
//
// Tunables. WARPS_PER_BLOCK is fixed at 8 (256/32). I2S_ROWS_PER_WARP rows are handled per warp; the
// block therefore covers I2S_ROWS_PER_BLOCK = 8 * I2S_ROWS_PER_WARP rows and stages x ONCE for all of
// them. Measured on the 3060 (sm_86): ROWS_PER_WARP=2 amortizes the half→float x stage over 16 rows and
// beats ROWS_PER_WARP=1 on the small-k attention shape (k=2560) while matching it on FFN (k=6912).
#define I2S_ROWS_PER_WARP  2
#define I2S_ROWS_PER_BLOCK (8 * I2S_ROWS_PER_WARP)   // 8 warps/block × rows-per-warp

#include <cuda_fp16.h>
#include <stdint.h>

// x[k] used to be staged into a fixed-size STATIC `__shared__ float xs[6912]` — sized for BitNet
// b1.58-2B-4T's largest per-tensor k (its FFN-down projection, k=intermediateSize=6912). That bound
// is architecture-specific, not a general I2_S contract: any non-BitNet Llama-body I2_S conversion
// whose intermediate size exceeds 6912 (e.g. Falcon-E-3B intermediate=13312, Falcon3-3B-Base
// intermediate=9216 — see issue #207) silently overflowed the static array on the FFN-down decode
// GEMV (`Project` dispatches Down through `i2_s_gemv_f16in` with k=DownInputDim=intermediateSize),
// corrupting adjacent shared memory and producing a CUDA "illegal memory access" fault. Every kernel
// below now uses DYNAMIC shared memory (`extern __shared__`), sized by the caller to `k *
// sizeof(float)` bytes (see `LaunchI2_SGemv*` in CudaKernels.cs, which also opts each function into
// the device's full dynamic-shared opt-in cap via `cuFuncSetAttribute`, mirroring the on-the-fly MMQ
// GEMV kernels' handling of the same class of bug).

// Intra-warp sum reduce (v2 warp-per-row path): the 32 lanes of one warp hold partial sums for a single
// output row; lane 0 ends with the total. No shared memory, no __syncthreads.
__device__ __forceinline__ float i2s_warp_reduce(float acc)
{
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    return acc;
}

// Decode the 4 codes packed in byte `p` (elements {gp,+32,+64,+96}) and accumulate into `acc`
// against the four shared activations at base `xb` + {0,32,64,96}. Branchless code-1 decode.
__device__ __forceinline__ void i2s_accum_byte(float& acc, unsigned int p, const float* xs, int xb)
{
    int c0 = ((p >> 6) & 0x3) - 1;
    int c1 = ((p >> 4) & 0x3) - 1;
    int c2 = ((p >> 2) & 0x3) - 1;
    int c3 = ( p       & 0x3) - 1;
    acc += (float)c0 * xs[xb];
    acc += (float)c1 * xs[xb + 32];
    acc += (float)c2 * xs[xb + 64];
    acc += (float)c3 * xs[xb + 96];
}

// Accumulate one uint4 (16 bytes of one 128-block) into `acc` from shared activations `xs`.
// `blkBase` = blk*128; `gp0` = byte-in-block of the uint4's first byte (0 or 16).
__device__ __forceinline__ void i2s_accum_u4(float& acc, uint4 w, const float* xs, int blkBase, int gp0)
{
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        unsigned int word = (&w.x)[j];      // 4 packed bytes
        int gpw = gp0 + j * 4;              // byte index in block of this word's byte 0
        i2s_accum_byte(acc, (word      ) & 0xFF, xs, blkBase + gpw    );
        i2s_accum_byte(acc, (word >>  8) & 0xFF, xs, blkBase + gpw + 1);
        i2s_accum_byte(acc, (word >> 16) & 0xFF, xs, blkBase + gpw + 2);
        i2s_accum_byte(acc, (word >> 24) & 0xFF, xs, blkBase + gpw + 3);
    }
}

// ───────────────────────── Variant A: W2A16, FP16 activations ─────────────────────────
extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_f16in(
    const uint8_t* __restrict__ weight,   // packed codes [n × k/4 bytes] + trailing f32 scale
    const half*    __restrict__ x,        // [k]
    half*          __restrict__ y,        // [n]
    const int n,
    const int k)
{
    // Stage x[k] into shared memory once per block (FP16 -> FP32), all 256 threads cooperating.
    // Reused by all I2S_ROWS_PER_BLOCK warps in this block. Dynamic — caller sizes to k*4 bytes.
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = __half2float(x[i]);
    __syncthreads();

    const int wid  = threadIdx.x >> 5;        // warp id within block (0..7)
    const int lane = threadIdx.x & 31;        // lane within warp

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));
    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;     // 16 bytes per uint4

    // Each warp owns I2S_ROWS_PER_WARP consecutive-in-block rows. v3 INTERLEAVES the rows inside ONE
    // u-loop instead of finishing one row before starting the next (v2): per iteration each lane issues
    // I2S_ROWS_PER_WARP independent uint4 weight loads (one per row) BEFORE the ALU-heavy decode, so
    // the rows' load streams overlap and hide global-load latency. The kernel is far from weight-DRAM
    // saturation (~5× headroom to the 3060's 360 GB/s) — it is latency/ILP-bound — so widening the
    // in-flight load window is the lever. Numerics unchanged (each row's fp32 sum order is identical).
    const int rowBase = blockIdx.x * I2S_ROWS_PER_BLOCK + wid * I2S_ROWS_PER_WARP;
    if (rowBase >= n) return;

    const uint4* w_rows[I2S_ROWS_PER_WARP];
    float acc[I2S_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        int r = min(rowBase + rr, n - 1);     // clamp tail rows; their result is discarded below
        w_rows[rr] = reinterpret_cast<const uint4*>(weight + (size_t)r * row_bytes);
        acc[rr] = 0.0f;
    }

    for (int u = lane; u < num_u4; u += warpSize)
    {
        int boff   = u << 4;
        int blkBase = (boff >> 5) << 7;       // (blk) * 128
        int gp0    = boff & 31;               // 0 or 16

        // Issue all rows' loads first → independent memory traffic overlaps.
        uint4 w[I2S_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) w[rr] = w_rows[rr][u];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) i2s_accum_u4(acc[rr], w[rr], xs, blkBase, gp0);
    }

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        float a = i2s_warp_reduce(acc[rr]);
        int row = rowBase + rr;
        if (lane == 0 && row < n) y[row] = __float2half(a * scale);
    }
}

// ───────────────────────── Variant A twin: FP32 activations ─────────────────────────
// Exact-match reference for CPU-vs-GPU validation and any F32 activation path.
extern "C" __global__ void __launch_bounds__(256) i2_s_gemv2_f16in(
    const uint8_t* __restrict__ weight0,
    const uint8_t* __restrict__ weight1,
    const half*    __restrict__ x,
    half*          __restrict__ y0,
    half*          __restrict__ y1,
    const int n0,
    const int n1,
    const int k)
{
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = __half2float(x[i]);
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int totalN = n0 + n1;
    const int rowBase = blockIdx.x * I2S_ROWS_PER_BLOCK + wid * I2S_ROWS_PER_WARP;
    if (rowBase >= totalN) return;

    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;

    const uint4* w_rows[I2S_ROWS_PER_WARP];
    half* y_rows[I2S_ROWS_PER_WARP];
    int local_rows[I2S_ROWS_PER_WARP];
    float scales[I2S_ROWS_PER_WARP];
    float acc[I2S_ROWS_PER_WARP];

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        int globalRow = min(rowBase + rr, totalN - 1);
        const uint8_t* w;
        int n;
        int localRow;
        if (globalRow < n0)
        {
            w = weight0;
            n = n0;
            y_rows[rr] = y0;
            localRow = globalRow;
        }
        else
        {
            w = weight1;
            n = n1;
            y_rows[rr] = y1;
            localRow = globalRow - n0;
        }

        local_rows[rr] = localRow;
        scales[rr] = *reinterpret_cast<const float*>(w + (size_t)n * row_bytes);
        w_rows[rr] = reinterpret_cast<const uint4*>(w + (size_t)localRow * row_bytes);
        acc[rr] = 0.0f;
    }

    for (int u = lane; u < num_u4; u += warpSize)
    {
        int boff    = u << 4;
        int blkBase = (boff >> 5) << 7;
        int gp0     = boff & 31;

        uint4 w[I2S_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) w[rr] = w_rows[rr][u];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) i2s_accum_u4(acc[rr], w[rr], xs, blkBase, gp0);
    }

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        float a = i2s_warp_reduce(acc[rr]);
        int globalRow = rowBase + rr;
        if (lane == 0 && globalRow < totalN)
            y_rows[rr][local_rows[rr]] = __float2half(a * scales[rr]);
    }
}

extern "C" __global__ void __launch_bounds__(256) i2_s_gemv3_f16in(
    const uint8_t* __restrict__ weight0,
    const uint8_t* __restrict__ weight1,
    const uint8_t* __restrict__ weight2,
    const half*    __restrict__ x,
    half*          __restrict__ y0,
    half*          __restrict__ y1,
    half*          __restrict__ y2,
    const int n0,
    const int n1,
    const int n2,
    const int k)
{
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = __half2float(x[i]);
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int totalN = n0 + n1 + n2;
    const int rowBase = blockIdx.x * I2S_ROWS_PER_BLOCK + wid * I2S_ROWS_PER_WARP;
    if (rowBase >= totalN) return;

    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;

    const uint4* w_rows[I2S_ROWS_PER_WARP];
    half* y_rows[I2S_ROWS_PER_WARP];
    int local_rows[I2S_ROWS_PER_WARP];
    float scales[I2S_ROWS_PER_WARP];
    float acc[I2S_ROWS_PER_WARP];

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        int globalRow = min(rowBase + rr, totalN - 1);
        const uint8_t* w;
        int n;
        int localRow;
        if (globalRow < n0)
        {
            w = weight0;
            n = n0;
            y_rows[rr] = y0;
            localRow = globalRow;
        }
        else if (globalRow < n0 + n1)
        {
            w = weight1;
            n = n1;
            y_rows[rr] = y1;
            localRow = globalRow - n0;
        }
        else
        {
            w = weight2;
            n = n2;
            y_rows[rr] = y2;
            localRow = globalRow - n0 - n1;
        }

        local_rows[rr] = localRow;
        scales[rr] = *reinterpret_cast<const float*>(w + (size_t)n * row_bytes);
        w_rows[rr] = reinterpret_cast<const uint4*>(w + (size_t)localRow * row_bytes);
        acc[rr] = 0.0f;
    }

    for (int u = lane; u < num_u4; u += warpSize)
    {
        int boff    = u << 4;
        int blkBase = (boff >> 5) << 7;
        int gp0     = boff & 31;

        uint4 w[I2S_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) w[rr] = w_rows[rr][u];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) i2s_accum_u4(acc[rr], w[rr], xs, blkBase, gp0);
    }

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        float a = i2s_warp_reduce(acc[rr]);
        int globalRow = rowBase + rr;
        if (lane == 0 && globalRow < totalN)
            y_rows[rr][local_rows[rr]] = __float2half(a * scales[rr]);
    }
}

extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_norm_f16in(
    const uint8_t* __restrict__ weight,
    const half*    __restrict__ x,
    const half*    __restrict__ norm_weight,
    half*          __restrict__ y,
    const int n,
    const int k,
    const float eps)
{
    // Dynamic shared layout: [0, k) = xs (RMS-normalized x, reused by the GEMV below),
    // [k_aligned, k_aligned+32) = warp-sum reduction scratch, [k_aligned+32] = rms_inv.
    // Caller sizes sharedBytes = (k + 33) * sizeof(float) (see LaunchI2_SGemvNormF16In).
    extern __shared__ float smem[];
    float* xs = smem;
    const int scratch_off = (k + 1) & ~1; // even-align, mirrors fused_add_rmsnorm.cu
    float* warp_sums = smem + scratch_off;
    float* rms_inv_ptr = warp_sums + 32;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        sum_sq += v * v;
    }

    for (int off = warpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);

    const int lane0 = threadIdx.x & 31;
    const int wid0 = threadIdx.x >> 5;
    if (lane0 == 0) warp_sums[wid0] = sum_sq;
    __syncthreads();

    if (wid0 == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane0 < num_warps) ? warp_sums[lane0] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
        if (lane0 == 0) *rms_inv_ptr = rsqrtf(sum_sq / (float)k + eps);
    }
    __syncthreads();

    const float rms_inv = *rms_inv_ptr;
    for (int i = threadIdx.x; i < k; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        float w = __half2float(norm_weight[i]);
        xs[i] = v * rms_inv * w;
    }
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));
    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;
    const int rowBase = blockIdx.x * I2S_ROWS_PER_BLOCK + wid * I2S_ROWS_PER_WARP;
    if (rowBase >= n) return;

    const uint4* w_rows[I2S_ROWS_PER_WARP];
    float acc[I2S_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        int r = min(rowBase + rr, n - 1);
        w_rows[rr] = reinterpret_cast<const uint4*>(weight + (size_t)r * row_bytes);
        acc[rr] = 0.0f;
    }

    for (int u = lane; u < num_u4; u += warpSize)
    {
        int boff    = u << 4;
        int blkBase = (boff >> 5) << 7;
        int gp0     = boff & 31;

        uint4 w[I2S_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) w[rr] = w_rows[rr][u];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) i2s_accum_u4(acc[rr], w[rr], xs, blkBase, gp0);
    }

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        float a = i2s_warp_reduce(acc[rr]);
        int row = rowBase + rr;
        if (lane == 0 && row < n) y[row] = __float2half(a * scale);
    }
}

extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_f32in(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    const int n,
    const int k)
{
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = x[i];
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));
    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;

    // Same row-interleaved ILP structure as the f16in twin (kept identical for exact CPU-parity).
    const int rowBase = blockIdx.x * I2S_ROWS_PER_BLOCK + wid * I2S_ROWS_PER_WARP;
    if (rowBase >= n) return;

    const uint4* w_rows[I2S_ROWS_PER_WARP];
    float acc[I2S_ROWS_PER_WARP];
    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        int r = min(rowBase + rr, n - 1);
        w_rows[rr] = reinterpret_cast<const uint4*>(weight + (size_t)r * row_bytes);
        acc[rr] = 0.0f;
    }

    for (int u = lane; u < num_u4; u += warpSize)
    {
        int boff    = u << 4;
        int blkBase = (boff >> 5) << 7;
        int gp0     = boff & 31;

        uint4 w[I2S_ROWS_PER_WARP];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) w[rr] = w_rows[rr][u];
        #pragma unroll
        for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++) i2s_accum_u4(acc[rr], w[rr], xs, blkBase, gp0);
    }

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
        float a = i2s_warp_reduce(acc[rr]);
        int row = rowBase + rr;
        if (lane == 0 && row < n) y[row] = a * scale;
    }
}

// ───────────────────────── Variant B: W2A8 (int8 activations, __dp4a) ─────────────────────────
//
// Spec: .planning/cuda-bitnet-gemv-prep.md §1, §3 Variant B, §7 row B0, §8.
//
// Technique borrowed from Microsoft BitNet's GPU W2A8 kernel: quantize activations per token to int8,
// decode ternary weights to int8, accumulate the integer dot product with __dp4a (4-way int8 dot +
// int32 accumulate), then apply the float epilogue. We adopt only the *technique* — NOT BitNet's
// packing, NOT its offset-by-2 code mapping. dotLLM's I2_S layout and "code-1" mapping are preserved.
//
// Occupancy/MLP optimization: all 256 threads stride over the row's uint4 units (16 B = 64 codes /
// load). The int8 activations are read directly from global (__ldg, L2-resident) — the single-byte
// shared gathers needed for the {gp,+32,+64,+96} interleave serialize and measured slower than L2,
// and the per-tensor xq is reused across all n blocks so it stays L2-hot.
//
// Activation contract (host- or kernel-quantized; tests quantize on host):
//   * x is quantized per token, symmetric absmax: s_act = 127 / absmax(x), xq_i = round(x_i * s_act).
//   * The kernel receives the int8 activations `xq[k]` plus `inv_act_scale = absmax(x)/127 = 1/s_act`,
//     so x_i ≈ xq_i * inv_act_scale.
//   * Output: out = scale * inv_act_scale * Σ_i ( xq_i · (code_i - 1) ).
//
// Decode: build the 4 ternary codes into an int8x4 register, subtract 1 across all lanes with
// __vsubss4 (saturating per-byte subtract — code-1 mapping, NOT BitNet's 0x02020202). Gather the 4
// matching int8 activations with the SAME lane order, then one __dp4a per byte. Requires sm_61+.
extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_a8(
    const uint8_t* __restrict__ weight,   // packed codes [n × k/4 bytes] + trailing f32 weight scale
    const int8_t*  __restrict__ xq,       // [k] int8 activations, per-token absmax-quantized
    float*         __restrict__ y,        // [n] fp32 output
    const int   n,
    const int   k,
    const float inv_act_scale)            // = absmax(x)/127 = 1/s_act ; x_i ≈ xq_i * inv_act_scale
{
    const int wid  = threadIdx.x >> 5;        // warp id (0..7)
    const int lane = threadIdx.x & 31;

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));
    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
    const int row = blockIdx.x * I2S_ROWS_PER_BLOCK + wid * I2S_ROWS_PER_WARP + rr;
    if (row >= n) return;

    const uint4* w_row  = reinterpret_cast<const uint4*>(weight + (size_t)row * row_bytes);

    int iacc = 0;
    for (int u = lane; u < num_u4; u += warpSize)
    {
        uint4 w = w_row[u];
        int boff    = u << 4;
        int blk     = boff >> 5;
        int gp0     = boff & 31;
        int blkBase = blk << 7;

        #pragma unroll
        for (int wi = 0; wi < 4; wi++)
        {
            unsigned int word = (&w.x)[wi];
            int gpw = gp0 + wi * 4;
            #pragma unroll
            for (int bi = 0; bi < 4; bi++)
            {
                unsigned int p  = (word >> (bi * 8)) & 0xFF;
                int xb = blkBase + gpw + bi;

                // Decode 4 codes {0,1,2} for elements {gp,+32,+64,+96} into an int8x4 register,
                // lane j = code(gp + 32*j); subtract 1 across all 4 lanes → {-1,0,+1}.
                unsigned int w_codes =
                      ((unsigned int)((p >> 6) & 0x3))
                    | ((unsigned int)((p >> 4) & 0x3) <<  8)
                    | ((unsigned int)((p >> 2) & 0x3) << 16)
                    | ((unsigned int)((p     ) & 0x3) << 24);
                int w_vec = __vsubss4((int)w_codes, 0x01010101);

                // Gather the 4 matching int8 activations with the SAME lane order from global (L2).
                unsigned int a_vec =
                      ((unsigned int)((unsigned char)__ldg(xq + xb      )))
                    | ((unsigned int)((unsigned char)__ldg(xq + xb +  32)) <<  8)
                    | ((unsigned int)((unsigned char)__ldg(xq + xb +  64)) << 16)
                    | ((unsigned int)((unsigned char)__ldg(xq + xb +  96)) << 24);

                iacc = __dp4a((int)a_vec, w_vec, iacc);
            }
        }
    }

    float acc = i2s_warp_reduce((float)iacc);
    if (lane == 0) y[row] = acc * scale * inv_act_scale;
    }
}

extern "C" __global__ void __launch_bounds__(256) quantize_f16_to_i8_absmax(
    const half* __restrict__ x,
    int8_t*     __restrict__ xq,
    float*      __restrict__ inv_act_scale,
    const int k)
{
    __shared__ float warp_max[32];
    __shared__ float scale;
    __shared__ float inv_scale;

    float local = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        local = fmaxf(local, fabsf(__half2float(x[i])));

    for (int off = warpSize / 2; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_down_sync(0xFFFFFFFF, local, off));

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_max[wid] = local;
    __syncthreads();

    if (wid == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local = lane < num_warps ? warp_max[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            local = fmaxf(local, __shfl_down_sync(0xFFFFFFFF, local, off));
        if (lane == 0)
        {
            float absmax = local > 0.0f ? local : 1.0f;
            inv_scale = absmax / 127.0f;
            scale = 127.0f / absmax;
            *inv_act_scale = inv_scale;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < k; i += blockDim.x)
    {
        float q = rintf(__half2float(x[i]) * scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        xq[i] = (int8_t)q;
    }
}

extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_a8_device_scale(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq,
    float*         __restrict__ y,
    const int   n,
    const int   k,
    const float* __restrict__ inv_act_scale)
{
    const float inv = *inv_act_scale;
    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));
    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;

    #pragma unroll
    for (int rr = 0; rr < I2S_ROWS_PER_WARP; rr++)
    {
    const int row = blockIdx.x * I2S_ROWS_PER_BLOCK + wid * I2S_ROWS_PER_WARP + rr;
    if (row >= n) return;

    const uint4* w_row  = reinterpret_cast<const uint4*>(weight + (size_t)row * row_bytes);

    int iacc = 0;
    for (int u = lane; u < num_u4; u += warpSize)
    {
        uint4 w = w_row[u];
        int boff    = u << 4;
        int blk     = boff >> 5;
        int gp0     = boff & 31;
        int blkBase = blk << 7;

        #pragma unroll
        for (int wi = 0; wi < 4; wi++)
        {
            unsigned int word = (&w.x)[wi];
            int gpw = gp0 + wi * 4;
            #pragma unroll
            for (int bi = 0; bi < 4; bi++)
            {
                unsigned int p  = (word >> (bi * 8)) & 0xFF;
                int xb = blkBase + gpw + bi;

                unsigned int w_codes =
                      ((unsigned int)((p >> 6) & 0x3))
                    | ((unsigned int)((p >> 4) & 0x3) <<  8)
                    | ((unsigned int)((p >> 2) & 0x3) << 16)
                    | ((unsigned int)((p     ) & 0x3) << 24);
                int w_vec = __vsubss4((int)w_codes, 0x01010101);

                unsigned int a_vec =
                      ((unsigned int)((unsigned char)__ldg(xq + xb      )))
                    | ((unsigned int)((unsigned char)__ldg(xq + xb +  32)) <<  8)
                    | ((unsigned int)((unsigned char)__ldg(xq + xb +  64)) << 16)
                    | ((unsigned int)((unsigned char)__ldg(xq + xb +  96)) << 24);

                iacc = __dp4a((int)a_vec, w_vec, iacc);
            }
        }
    }

    float acc = i2s_warp_reduce((float)iacc);
    if (lane == 0) y[row] = acc * scale * inv;
    }
}
