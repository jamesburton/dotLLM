#include <math.h>

// Each channel is independent; consume the complete chunk before rolling its state.
extern "C" __global__ void lfm2_shortconv_f32(float* state, const float* proj, const float* weights, float* out, int taps, int hidden, int tokens) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    for (int t = 0; t < tokens; ++t) {
        float sum = 0.f;
        for (int j = 0; j < taps; ++j) {
            int row = t + j - (taps - 1);
            float x = row < 0 ? state[(row + taps - 1) * hidden + c] : proj[row * 3 * hidden + c] * proj[row * 3 * hidden + 2 * hidden + c];
            sum += weights[j * hidden + c] * x;
        }
        out[t * hidden + c] = sum * proj[t * 3 * hidden + hidden + c];
    }
    for (int j = 0; j < taps - 1; ++j) {
        int row = tokens + j - (taps - 1);
        state[j * hidden + c] = row < 0 ? state[(tokens + j) * hidden + c] : proj[row * 3 * hidden + c] * proj[row * 3 * hidden + 2 * hidden + c];
    }
}

// The correction bias affects selection only; gathered weights use unbiased sigmoid scores.
// One lane per token: E=64, K=4 is tiny beside expert GEMMs, with deterministic index ties.
extern "C" __global__ void lfm2_sigmoid_topk_f32(const float* logits, const float* bias, int* indices, float* weights, int experts, int topk, int tokens) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;
    float total = 0.f;
    for (int s = 0; s < topk; ++s) {
        int best = -1; float bestScore = -__int_as_float(0x7f800000); float bestWeight = 0.f;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int j = 0; j < s; ++j) if (indices[t * topk + j] == e) used = true;
            if (used) continue;
            float v = 1.f / (1.f + expf(-logits[t * experts + e]));
            float score = v + (bias ? bias[e] : 0.f);
            if (best < 0 || score > bestScore) { best = e; bestScore = score; bestWeight = v; }
        }
        indices[t * topk + s] = best;
        weights[t * topk + s] = bestWeight;
        total += bestWeight;
    }
    if (total > 0.f) for (int s = 0; s < topk; ++s) weights[t * topk + s] /= total;
}
#include <cuda_fp16.h>
extern "C" __global__ void lfm2_embedding_q4_0_f32(const unsigned char* weights, const int* ids, float* output, int hidden, int tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden * tokens) return;
    int t = i / hidden, c = i % hidden;
    const unsigned char* block = weights + ((long long)ids[t] * (hidden / 32) + c / 32) * 18;
    int q = (block[2 + (c % 16)] >> (c % 32 < 16 ? 0 : 4)) & 15;
    output[i] = __half2float(*reinterpret_cast<const __half*>(block)) * (q - 8);
}
// Four output rows per block; one warp reduces each row directly from packed weights.
extern "C" __global__ void lfm2_gemv_q4_0_f32(const unsigned char* weights, const float* input, float* output, int rows, int cols) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    float sum = 0.f;
    for (int b = 0; b < cols / 32; ++b) {
        const unsigned char* block = weights + ((long long)row * (cols / 32) + b) * 18;
        int q = (block[2 + lane % 16] >> (lane < 16 ? 0 : 4)) & 15;
        sum += (__half2float(*reinterpret_cast<const __half*>(block)) * (q - 8)) * input[b * 32 + lane];
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) output[row] = sum;
}
// GGML Q8_1 activation blocks: Half d, Half s=d_full*sum(q), then 32 signed bytes.
// d/s round separately, which matters for the Q4_0 zero-point correction.
extern "C" __global__ void lfm2_quant_q8_1_f32(const float* input, unsigned char* output, int blocks) {
    int lane = threadIdx.x % 32, block = blockIdx.x * 4 + threadIdx.x / 32;
    if (block >= blocks) return;
    float x = input[block * 32 + lane], maxAbs = fabsf(x);
    for (int offset = 16; offset > 0; offset >>= 1) maxAbs = fmaxf(maxAbs, __shfl_xor_sync(0xffffffff, maxAbs, offset));
    float scale = __fdiv_rn(maxAbs, 127.f);
    int q = scale == 0.f ? 0 : __float2int_rn(x * __fdiv_rn(1.f, scale));
    q = min(127, max(-127, q));
    reinterpret_cast<signed char*>(output + block * 36 + 4)[lane] = (signed char)q;
    int sum = q;
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) {
        *reinterpret_cast<__half*>(output + block * 36) = __float2half_rn(scale);
        *reinterpret_cast<__half*>(output + block * 36 + 2) = __float2half_rn(scale * sum);
    }
}
extern "C" __global__ void lfm2_gemm_q4_0_q8_1(const unsigned char* weights, const unsigned char* input, float* output, int rows, int cols) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32, token = blockIdx.y;
    if (row >= rows) return;
    float sum = 0.f, correction = 0.f;
    for (int b = 0; b < cols / 32; ++b) {
        const unsigned char* w = weights + ((long long)row * (cols / 32) + b) * 18;
        const unsigned char* x = input + ((long long)token * (cols / 32) + b) * 36;
        int nibble = (w[2 + lane % 16] >> (lane < 16 ? 0 : 4)) & 15;
        int dot = nibble * reinterpret_cast<const signed char*>(x + 4)[lane];
        for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0) {
            float d = __half2float(*reinterpret_cast<const __half*>(w));
            float d8 = __half2float(*reinterpret_cast<const __half*>(x));
            float s8 = __half2float(*reinterpret_cast<const __half*>(x + 2));
            sum += (d * d8) * dot;
            correction += d * s8;
        }
    }
    if (lane == 0) output[(long long)token * rows + row] = sum - 8.f * correction;
}
