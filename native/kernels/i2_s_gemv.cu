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
// ───────────────────────────── Occupancy / MLP optimization ─────────────────────────────
// The launch config is FIXED by CudaKernels.cs: grid = (n,1,1), block = (256,1,1), shared = 0.
// One block still owns exactly one output row. The original kernel handed each thread one
// 128-element block (32 bytes), so a row with only k/128 = 20..54 blocks left 200+ of the 256
// threads idle and issued only ~20..54 in-flight loads → ~3% of the 3060's bandwidth (~10-12 GB/s).
//
// This version keeps grid=n/block=256 but raises active-warp density and memory-level parallelism:
//   (a) ALL 256 threads stride over the row's packed bytes (k/4 = 640..1728 bytes ≫ 256), so every
//       thread issues loads — full warp occupancy instead of ~20..54 active lanes.
//   (b) Each thread reads weights as uint4 (16 B = 64 codes) per load instead of byte-by-byte,
//       giving wide, fully-coalesced 16-byte transactions across the warp (ld.global.nc.v4.u32).
//   (c) For the FP16/FP32 activation variants, x[k] is staged once into static shared memory by the
//       whole block, then read from shared inside the hot loop. This removes the repeated half→float
//       L2 traffic and measured fastest for the W2A16 decode path that the forward pass dispatches.
//       The int8/dp4a variant reads xq directly from global (__ldg, L2-resident): its single-byte
//       shared gathers serialize and measured *slower* than the L2 path, while the per-tensor xq is
//       reused across all n blocks so it stays L2-hot anyway.
// Numerics are unchanged from the original Variant A (exact ternary decode, fp32/int32 accumulate);
// only the fp32 reduction order shifts, which stays inside the test's <=1e-3 tolerance (measured
// max abs diff ~1e-6 vs the CPU float reference).
//
// Layout note for the uint4 load: row_bytes = k/4 is a multiple of 32 (k%128==0), hence a multiple
// of 16, so it splits cleanly into uint4 (16-byte) units. A uint4 spans bytes [16u, 16u+15]; since
// 16u is a multiple of 16, those 16 bytes lie inside a single 32-byte (128-element) block — blk and
// the x base address are constant across the uint4, only gp = byte index within the block varies.

#include <cuda_fp16.h>
#include <stdint.h>

// Largest k across BitNet 2B4T projections is 6912 (FFN down). Static shared x buffer is sized for
// that; the launch passes shared=0 so we cannot use dynamic shared memory. 6912 floats = 27 KB,
// under sm_86's 48 KB static cap.
#define I2S_MAX_K 6912

__device__ __forceinline__ float i2s_block_reduce(float acc)
{
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    }
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
    int row = blockIdx.x;
    if (row >= n) return;

    // Stage x[k] into shared memory once per block (FP16 -> FP32), all threads cooperating.
    __shared__ float xs[I2S_MAX_K];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = __half2float(x[i]);
    __syncthreads();

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;     // 16 bytes per uint4
    const uint4* w_row  = reinterpret_cast<const uint4*>(weight + (size_t)row * row_bytes);

    float acc = 0.0f;
    // Every thread strides over the row's uint4 units → all 256 threads issue 16-byte loads.
    for (int u = threadIdx.x; u < num_u4; u += blockDim.x)
    {
        uint4 w = w_row[u];
        int boff = u << 4;                    // byte offset of this uint4
        int blk  = boff >> 5;                 // 32 bytes per 128-block
        int gp0  = boff & 31;                 // 0 or 16
        i2s_accum_u4(acc, w, xs, blk << 7, gp0);
    }

    acc = i2s_block_reduce(acc);
    if (threadIdx.x == 0) y[row] = __float2half(acc * scale);
}

// ───────────────────────── Variant A twin: FP32 activations ─────────────────────────
// Exact-match reference for CPU-vs-GPU validation and any F32 activation path.
extern "C" __global__ void __launch_bounds__(256) i2_s_gemv_f32in(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    const int n,
    const int k)
{
    int row = blockIdx.x;
    if (row >= n) return;

    __shared__ float xs[I2S_MAX_K];
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        xs[i] = x[i];
    __syncthreads();

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;
    const uint4* w_row  = reinterpret_cast<const uint4*>(weight + (size_t)row * row_bytes);

    float acc = 0.0f;
    for (int u = threadIdx.x; u < num_u4; u += blockDim.x)
    {
        uint4 w = w_row[u];
        int boff = u << 4;
        int blk  = boff >> 5;
        int gp0  = boff & 31;
        i2s_accum_u4(acc, w, xs, blk << 7, gp0);
    }

    acc = i2s_block_reduce(acc);
    if (threadIdx.x == 0) y[row] = acc * scale;
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
    int row = blockIdx.x;
    if (row >= n) return;

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int num_u4    = row_bytes >> 4;
    const uint4* w_row  = reinterpret_cast<const uint4*>(weight + (size_t)row * row_bytes);

    int iacc = 0;
    for (int u = threadIdx.x; u < num_u4; u += blockDim.x)
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

    float acc = i2s_block_reduce((float)iacc);
    if (threadIdx.x == 0) y[row] = acc * scale * inv_act_scale;
}
