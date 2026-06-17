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
// Variant A (W2A16): decode ternary, multiply by half/float activations, fp32 accumulate.
// Numerically matches the CPU float reference (MatMul.GemvI2_S) to fp32 rounding.
// Control structure (grid=n, block=256, thread-stride over 128-blocks, two-stage warp reduction)
// is identical to quantized_gemv_q8_0.

#include <cuda_fp16.h>
#include <stdint.h>

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

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int blocks    = k / 128;
    const uint8_t* w_row = weight + (size_t)row * row_bytes;

    float acc = 0.0f;
    for (int blk = threadIdx.x; blk < blocks; blk += blockDim.x)
    {
        const uint8_t* bp = w_row + blk * 32;
        const int x_base  = blk * 128;
        #pragma unroll 8
        for (int gp = 0; gp < 32; gp++)
        {
            uint8_t p = bp[gp];
            int c0 = ((p >> 6) & 0x3) - 1;
            int c1 = ((p >> 4) & 0x3) - 1;
            int c2 = ((p >> 2) & 0x3) - 1;
            int c3 = ( p       & 0x3) - 1;
            acc += (float)c0 * __half2float(x[x_base + gp]);
            acc += (float)c1 * __half2float(x[x_base + gp + 32]);
            acc += (float)c2 * __half2float(x[x_base + gp + 64]);
            acc += (float)c3 * __half2float(x[x_base + gp + 96]);
        }
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

    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int blocks    = k / 128;
    const uint8_t* w_row = weight + (size_t)row * row_bytes;

    float acc = 0.0f;
    for (int blk = threadIdx.x; blk < blocks; blk += blockDim.x)
    {
        const uint8_t* bp = w_row + blk * 32;
        const int x_base  = blk * 128;
        #pragma unroll 8
        for (int gp = 0; gp < 32; gp++)
        {
            uint8_t p = bp[gp];
            int c0 = ((p >> 6) & 0x3) - 1;
            int c1 = ((p >> 4) & 0x3) - 1;
            int c2 = ((p >> 2) & 0x3) - 1;
            int c3 = ( p       & 0x3) - 1;
            acc += (float)c0 * x[x_base + gp];
            acc += (float)c1 * x[x_base + gp + 32];
            acc += (float)c2 * x[x_base + gp + 64];
            acc += (float)c3 * x[x_base + gp + 96];
        }
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
// Activation contract (host- or kernel-quantized; tests quantize on host):
//   * x is quantized per token, symmetric absmax: s_act = 127 / absmax(x), xq_i = round(x_i * s_act).
//   * The kernel receives the int8 activations `xq[k]` plus `inv_act_scale = absmax(x)/127 = 1/s_act`,
//     so x_i ≈ xq_i * inv_act_scale. Passing the *inverse* scale keeps the epilogue a single multiply
//     and avoids a divide in the kernel.
//   * Output: out = scale * inv_act_scale * Σ_i ( xq_i · (code_i - 1) ).
//     (= weight_scale * (1/s_act) * integer_dot — matches BitNet's `acc / s_act * weight_scale`.)
//
// dp4a layout note (decision: B1, no layout repack).
//   dotLLM packs byte `gp` with elements {gp, gp+32, gp+64, gp+96} (bit offsets {6,4,2,0}). Those four
//   activations are NOT contiguous in x, so a plain `int4`/`int8x4` activation load would gather the
//   wrong lanes. B1 keeps the on-disk layout untouched and instead builds two matching int8x4 registers
//   per byte:
//     - w_vec : the 4 decoded ternary codes (code-1) for elements {gp, gp+32, gp+64, gp+96}.
//     - a_vec : the 4 int8 activations xq[base+gp+{0,32,64,96}] gathered with the SAME stride.
//   One __dp4a(a_vec, w_vec, acc) then replaces the 4 scalar FMAs of Variant A. The decode is a single
//   __vsubss4 (subtract 1 — code-1 mapping, NOT BitNet's 0x02020202). The lane ordering inside the two
//   int8x4 registers is identical (lane j ↔ element gp + 32*j for both), so __dp4a pairs them correctly
//   regardless of byte endianness — we build both vectors with the same packing so any consistent lane
//   numbering yields the right pairing.
//   (Alternative B2 — an upload-time repack into "4 contiguous codes per byte" so a plain contiguous
//   int8x4 activation load works — was rejected: it would require changing the weight-upload path and
//   keeping a divergent on-device layout. B1 needs no layout change.)
//
// Grid / block / reduction: identical to Variant A (grid=n, block=256, thread-stride over 128-blocks,
// i2s_block_reduce two-stage warp reduction). Requires sm_61+ for __dp4a (dotLLM builds compute_61).
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

    // Per-tensor weight scale at the tensor tail: byte offset (size_t)n*(k/4).
    const float scale = *reinterpret_cast<const float*>(weight + (size_t)n * (k / 4));

    const int row_bytes = k / 4;
    const int blocks    = k / 128;
    const uint8_t* w_row = weight + (size_t)row * row_bytes;

    // Integer accumulator: exact int32 dot of int8 activations with ternary codes. No rounding here —
    // all approximation lives in the host activation quant, so this matches the int8 CPU reference.
    int iacc = 0;

    for (int blk = threadIdx.x; blk < blocks; blk += blockDim.x)
    {
        const uint8_t* bp = w_row + blk * 32;
        const int x_base  = blk * 128;
        #pragma unroll 8
        for (int gp = 0; gp < 32; gp++)
        {
            uint8_t p = bp[gp];

            // Decode 4 codes {0,1,2} for elements {gp, gp+32, gp+64, gp+96} into an int8x4 register,
            // lane j (byte j) = code(gp + 32*j). Build as raw codes, then subtract 1 across all 4 lanes
            // with __vsubss4 (saturating per-byte subtract) → {-1,0,+1}. This is the code-1 mapping.
            unsigned int w_codes =
                  ((unsigned int)((p >> 6) & 0x3))        // lane 0 → element gp
                | ((unsigned int)((p >> 4) & 0x3) <<  8)  // lane 1 → element gp+32
                | ((unsigned int)((p >> 2) & 0x3) << 16)  // lane 2 → element gp+64
                | ((unsigned int)((p     ) & 0x3) << 24); // lane 3 → element gp+96
            int w_vec = __vsubss4((int)w_codes, 0x01010101);   // {0,1,2} → {-1,0,+1}

            // Gather the 4 matching int8 activations with the SAME lane order: lane j = xq[base+gp+32*j].
            // Mask to a byte each so the assembled word is a clean int8x4 (xq is signed int8).
            unsigned int a_vec =
                  ((unsigned int)((unsigned char)xq[x_base + gp      ]))
                | ((unsigned int)((unsigned char)xq[x_base + gp +  32]) <<  8)
                | ((unsigned int)((unsigned char)xq[x_base + gp +  64]) << 16)
                | ((unsigned int)((unsigned char)xq[x_base + gp +  96]) << 24);

            iacc = __dp4a((int)a_vec, w_vec, iacc);   // Σ_lane (a_lane * w_lane), int32 accumulate
        }
    }

    // fp32-accumulate the per-thread int32 partials across the block, then scale once.
    float acc = i2s_block_reduce((float)iacc);
    if (threadIdx.x == 0) y[row] = acc * scale * inv_act_scale;
}
