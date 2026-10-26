// TurboQuant (MSE stage) KV codec kernels for dotLLM — the CUDA port of the Vulkan
// turboquant_{dequant,encode}_f32.comp shaders and DotLLM.Engine.KvCache.Codecs.TurboQuantCodec.
//
// Per cached head-vector (dimension headDim, a power of two <= 256):
//   codes: headDim packed indices of `mseBits` bits each, LSB-first within a vector, coord i at
//          bit offset i*mseBits; tightly packed across head-vectors at a uint-aligned stride
//          codeUintsPerVec = ceil((headDim*mseBits + 7)/8 / 4) uints per vector.
//   norm:  one fp32 L2 norm per head-vector.
// Decode:  x[i]=centroids[code_i]; x=WalshHadamard(x); x[i]*=invSqrtD*signs[i]*norm.
// Encode:  norm; unit=x/norm; s=(1/sqrt d) H (D unit); code_i=argmin_j |s_i - centroids[j]|; pack.
// centroids[] (already scaled by 1/sqrt d), signs[] (+/-1) and invSqrtD are codec constants.
//
// Layout: one CUDA block per head-vector, blockDim.x = 256 (one thread per coordinate; threads
// with t >= headDim stay live but idle so every thread reaches each __syncthreads()). F32 in/out
// to match the Vulkan reference and allow a bit-for-bit comparison against the CPU codec.

#include <stdint.h>
#include <cuda_fp16.h>

#define TQ_MAX_HEADDIM 256

// ── Dequant: codes -> fp32 scratch ──────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256) turboquant_dequant_f32(
    const uint32_t* __restrict__ codes,
    const float*    __restrict__ norms,
    const float*    __restrict__ centroids,
    const float*    __restrict__ signs,
    float*          __restrict__ dst,
    const int numVectors, const int headDim, const int numKvHeads,
    const int mseBits, const int codeUintsPerVec, const float invSqrtD)
{
    __shared__ float s[TQ_MAX_HEADDIM];

    const int hv = blockIdx.x;
    if (hv >= numVectors) return;
    const int t = threadIdx.x;
    const int d = headDim;
    const bool live = t < d;

    // 1. centroid lookup: read mseBits bits at bit offset t*mseBits within this vector.
    if (live)
    {
        const unsigned baseBit = (unsigned)hv * (unsigned)codeUintsPerVec * 32u;
        const unsigned p = baseBit + (unsigned)t * (unsigned)mseBits;
        const unsigned widx = p >> 5;
        const unsigned boff = p & 31u;
        const unsigned mask = (mseBits >= 32) ? 0xFFFFFFFFu : ((1u << mseBits) - 1u);
        unsigned val = codes[widx] >> boff;
        if (boff + (unsigned)mseBits > 32u)
            val |= codes[widx + 1] << (32u - boff);
        val &= mask;
        s[t] = centroids[val];
    }
    __syncthreads();

    // 2. in-place unnormalized Walsh-Hadamard transform (cooperative butterfly).
    for (int len = 1; len < d; len <<= 1)
    {
        if (live && (t & len) == 0)
        {
            const float u = s[t];
            const float v = s[t + len];
            s[t]       = u + v;
            s[t + len] = u - v;
        }
        __syncthreads();
    }

    // 3. inverse rotation + rescale by the stored norm.
    if (live)
    {
        const float outv = s[t] * invSqrtD * signs[t] * norms[hv];
        const int pos    = hv / numKvHeads;
        const int h      = hv - pos * numKvHeads;
        const int stride = numKvHeads * d;
        dst[(size_t)pos * stride + (size_t)h * d + t] = outv;
    }
}

// ── Encode: fresh fp32 K/V -> codes + norm ──────────────────────────────────
// Source rows are contiguous [seqLen, numKvHeads*headDim]; block b -> srcRow=b/numKvHeads,
// h=b%numKvHeads; destination head-vector index is (startPos+srcRow)*numKvHeads+h. Each block owns
// its (disjoint, uint-aligned) code uints, so output is single-writer (no atomics/pre-zeroing).
extern "C" __global__ void __launch_bounds__(256) turboquant_encode_f32(
    const float* __restrict__ src,
    const float* __restrict__ centroids,
    const float* __restrict__ signs,
    uint32_t*    __restrict__ codes,
    float*       __restrict__ norms,
    const int headDim, const int numKvHeads, const int mseBits, const int codeUintsPerVec,
    const int levelCount, const int startPos, const float invSqrtD)
{
    __shared__ float sh[TQ_MAX_HEADDIM];
    __shared__ unsigned idx[TQ_MAX_HEADDIM];

    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int d = headDim;
    const bool live = t < d;

    const int srcRow = b / numKvHeads;
    const int h      = b - srcRow * numKvHeads;
    const int stride = numKvHeads * d;
    const size_t srcOff = (size_t)srcRow * stride + (size_t)h * d;

    // norm = ||x|| via shared tree reduction.
    const float xt = live ? src[srcOff + t] : 0.0f;
    sh[t] = xt * xt;
    __syncthreads();
    for (int r = d >> 1; r > 0; r >>= 1)
    {
        if (t < r) sh[t] += sh[t + r];
        __syncthreads();
    }
    const float norm = sqrtf(sh[0]);
    __syncthreads();

    // unit direction -> forward RHT (sign-flip, WHT, normalize).
    const float unit = (norm > 0.0f && live) ? (xt / norm) : 0.0f;
    sh[t] = live ? unit * signs[t] : 0.0f;
    __syncthreads();
    for (int len = 1; len < d; len <<= 1)
    {
        if (live && (t & len) == 0)
        {
            const float u = sh[t];
            const float v = sh[t + len];
            sh[t]       = u + v;
            sh[t + len] = u - v;
        }
        __syncthreads();
    }

    // nearest centroid (ascending; argmin |s - c|).
    if (live)
    {
        const float val = sh[t] * invSqrtD;
        unsigned best = 0;
        float bestD = fabsf(val - centroids[0]);
        for (int j = 1; j < levelCount; j++)
        {
            const float dist = fabsf(val - centroids[j]);
            if (dist < bestD) { bestD = dist; best = (unsigned)j; }
        }
        idx[t] = best;
    }
    __syncthreads();

    // assemble code uints (one writer per uint: this block owns them).
    const unsigned vecHv = (unsigned)(startPos + srcRow) * (unsigned)numKvHeads + (unsigned)h;
    const unsigned vecBaseUint = vecHv * (unsigned)codeUintsPerVec;
    if (t < codeUintsPerVec)
    {
        unsigned word = 0u;
        const unsigned wordLoBit = (unsigned)t * 32u;
        const unsigned wordHiBit = wordLoBit + 32u;
        unsigned cStart = wordLoBit / (unsigned)mseBits;
        for (unsigned c = cStart; c < (unsigned)d; c++)
        {
            const unsigned cBit = c * (unsigned)mseBits;
            if (cBit >= wordHiBit) break;
            const unsigned code = idx[c];
            if (cBit >= wordLoBit) word |= code << (cBit - wordLoBit);
            else                   word |= code >> (wordLoBit - cBit);
        }
        codes[vecBaseUint + t] = word;
    }

    if (t == 0) norms[vecHv] = norm;
}

// ── FP16 variants for the CUDA forward (K/V activations + attention scratch are half) ──────────
// Identical math to the F32 kernels; only the K/V data I/O is half. Codes (uint), norm (fp32) and
// the codec constants (centroids/signs, fp32) are unchanged.

extern "C" __global__ void __launch_bounds__(256) turboquant_dequant_f16(
    const uint32_t* __restrict__ codes,
    const float*    __restrict__ norms,
    const float*    __restrict__ centroids,
    const float*    __restrict__ signs,
    half*           __restrict__ dst,
    const int numVectors, const int headDim, const int numKvHeads,
    const int mseBits, const int codeUintsPerVec, const float invSqrtD)
{
    __shared__ float s[TQ_MAX_HEADDIM];
    const int hv = blockIdx.x;
    if (hv >= numVectors) return;
    const int t = threadIdx.x;
    const int d = headDim;
    const bool live = t < d;

    if (live)
    {
        const unsigned baseBit = (unsigned)hv * (unsigned)codeUintsPerVec * 32u;
        const unsigned p = baseBit + (unsigned)t * (unsigned)mseBits;
        const unsigned widx = p >> 5;
        const unsigned boff = p & 31u;
        const unsigned mask = (mseBits >= 32) ? 0xFFFFFFFFu : ((1u << mseBits) - 1u);
        unsigned val = codes[widx] >> boff;
        if (boff + (unsigned)mseBits > 32u) val |= codes[widx + 1] << (32u - boff);
        val &= mask;
        s[t] = centroids[val];
    }
    __syncthreads();
    for (int len = 1; len < d; len <<= 1)
    {
        if (live && (t & len) == 0) { const float u = s[t]; const float v = s[t + len]; s[t] = u + v; s[t + len] = u - v; }
        __syncthreads();
    }
    if (live)
    {
        const float outv = s[t] * invSqrtD * signs[t] * norms[hv];
        const int pos    = hv / numKvHeads;
        const int h      = hv - pos * numKvHeads;
        const int stride = numKvHeads * d;
        dst[(size_t)pos * stride + (size_t)h * d + t] = __float2half(outv);
    }
}

extern "C" __global__ void __launch_bounds__(256) turboquant_encode_f16(
    const half*  __restrict__ src,
    const float* __restrict__ centroids,
    const float* __restrict__ signs,
    uint32_t*    __restrict__ codes,
    float*       __restrict__ norms,
    const int headDim, const int numKvHeads, const int mseBits, const int codeUintsPerVec,
    const int levelCount, const int startPos, const float invSqrtD)
{
    __shared__ float sh[TQ_MAX_HEADDIM];
    __shared__ unsigned idx[TQ_MAX_HEADDIM];

    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int d = headDim;
    const bool live = t < d;

    const int srcRow = b / numKvHeads;
    const int h      = b - srcRow * numKvHeads;
    const int stride = numKvHeads * d;
    const size_t srcOff = (size_t)srcRow * stride + (size_t)h * d;

    const float xt = live ? __half2float(src[srcOff + t]) : 0.0f;
    sh[t] = xt * xt;
    __syncthreads();
    for (int r = d >> 1; r > 0; r >>= 1) { if (t < r) sh[t] += sh[t + r]; __syncthreads(); }
    const float norm = sqrtf(sh[0]);
    __syncthreads();

    const float unit = (norm > 0.0f && live) ? (xt / norm) : 0.0f;
    sh[t] = live ? unit * signs[t] : 0.0f;
    __syncthreads();
    for (int len = 1; len < d; len <<= 1)
    {
        if (live && (t & len) == 0) { const float u = sh[t]; const float v = sh[t + len]; sh[t] = u + v; sh[t + len] = u - v; }
        __syncthreads();
    }
    if (live)
    {
        const float val = sh[t] * invSqrtD;
        unsigned best = 0; float bestD = fabsf(val - centroids[0]);
        for (int j = 1; j < levelCount; j++) { const float dist = fabsf(val - centroids[j]); if (dist < bestD) { bestD = dist; best = (unsigned)j; } }
        idx[t] = best;
    }
    __syncthreads();

    const unsigned vecHv = (unsigned)(startPos + srcRow) * (unsigned)numKvHeads + (unsigned)h;
    const unsigned vecBaseUint = vecHv * (unsigned)codeUintsPerVec;
    if (t < codeUintsPerVec)
    {
        unsigned word = 0u;
        const unsigned wordLoBit = (unsigned)t * 32u;
        const unsigned wordHiBit = wordLoBit + 32u;
        unsigned cStart = wordLoBit / (unsigned)mseBits;
        for (unsigned c = cStart; c < (unsigned)d; c++)
        {
            const unsigned cBit = c * (unsigned)mseBits;
            if (cBit >= wordHiBit) break;
            const unsigned code = idx[c];
            if (cBit >= wordLoBit) word |= code << (cBit - wordLoBit);
            else                   word |= code >> (wordLoBit - cBit);
        }
        codes[vecBaseUint + t] = word;
    }
    if (t == 0) norms[vecHv] = norm;
}
