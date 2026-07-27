// Gemma-4 (DiffusionGemma AR) F32 helper kernels.
//
// These cover the ops the gemma4 MoE forward needs that the existing F32
// kernel catalog does not provide:
//   * geglu_tanh_f32          — GeGLU with the tanh-approx GELU (dense + experts).
//   * rope_f32_partial_neox   — partial NeoX RoPE: rotate the leading `rotated_pairs`
//                               PAIRS of each head, pairing (i, i + head_dim/2) over
//                               the FULL head dim, freq base over the full head dim
//                               (Gemma-4 global layers: n_rot=512, only 64 pairs
//                               rotate, dims [0,64) <-> [256,320)).
//   * scale_inplace_f32       — in-place scalar multiply (layer_output_scale).
//   * rmsnorm_weightless_f32  — per-row RMSNorm with NO learned weight (unit gamma);
//                               used for the weight-less V-norm (one row per kv head).
//   * softcap_inplace_f32     — final-logit soft-capping: c * tanh(x / c).
//
// All F32 in/out to match the CPU oracle algorithmically (drift only from GPU
// reduction order). Authoritative semantics: docs/diffusiongemma/GEMMA4-GRAPH-SPEC.md
// and the CPU reference TransformerModel.RunGemma4Layer / RoPE.ExecutePartialNeoX.

#include <math.h>
#include <cuda_fp16.h>

// ── Activation Q8_0 round-trip (FP32 in/out) ──
// Reproduces the CPU oracle's on-the-fly ACTIVATION quantization for Q8_0-weight
// GEMMs (MatMul.GemmQ8_0 quantizes the F32 activation to Q8_0 before the int8
// dot). The CUDA gemma4 forward runs F32 cuBLAS on a dequantized weight, which
// otherwise keeps the activation in full F32 — so the two backends do slightly
// DIFFERENT math and drift (~1.7e-3/op, compounding across layers until the
// worst logit pokes over the parity tolerance). One warp per 32-block.
//
// Per 32-element block: scale = (half)(maxabs / 127); q = round_nearest_even(x/scale)
// clamped to ±127; x_out = q * (float)scale. The scale is stored/consumed as FP16
// exactly like the CPU's Q8_0 block scale (Half). The last partial group
// (k % 32 != 0) is passed through unchanged (matches the CPU, which only
// quantizes whole 32-blocks; gemma4 K dims are always %32).
extern "C" __global__ void __launch_bounds__(32) quantize_activation_q8_0_roundtrip_f32(
    float* __restrict__ x, const int k, const int rows)
{
    int nb = k / 32;
    long total = (long)rows * nb;
    int gid = blockIdx.x;                 // one warp per 32-block
    if (gid >= total) return;

    int row = gid / nb;
    int blk = gid % nb;
    float* base = x + (size_t)row * k + (size_t)blk * 32;
    int lane = threadIdx.x;               // 0..31

    float v = base[lane];
    float a = fabsf(v);
    // Warp max-abs reduction (every lane ends with the block max-abs).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, off));
    float scale_f = __half2float(__float2half(a / 127.0f));   // FP16 scale, like CPU
    if (scale_f == 0.0f) { base[lane] = 0.0f; return; }
    float inv = 1.0f / scale_f;
    int q = __float2int_rn(v * inv);      // round-nearest-even, matches CPU
    q = max(-127, min(127, q));
    base[lane] = (float)q * scale_f;
}

// ── tanh-approx GELU GeGLU: output = gelu_tanh(gate) * up ──
// gelu_tanh(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
// Matches DotLLM.Cpu.Kernels.FusedOps.GeGLUTanh / ggml_geglu (GGML_GLU_OP_GEGLU).
extern "C" __global__ void __launch_bounds__(256) geglu_tanh_f32(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, const int n, const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * seq_len)
    {
        const float kBeta  = 0.7978845608028654f;  // sqrt(2/pi)
        const float kAlpha = 0.044715f;
        float x = gate[idx];
        float inner = kBeta * (x + kAlpha * x * x * x);
        float g = 0.5f * x * (1.0f + tanhf(inner));
        output[idx] = g * up[idx];
    }
}

// ── Partial NeoX RoPE (FP32) ──
// Rotates the leading `rotated_pairs` pairs of each head. Pair i couples
// (vec[i], vec[i + head_dim/2]); freq base over the FULL head_dim. Dims beyond
// the rotated span pass through unchanged. NeoX-only (Gemma pairing).
extern "C" __global__ void __launch_bounds__(256) rope_f32_partial_neox(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    const int seq_len, const int num_heads, const int num_kv_heads,
    const int head_dim, const int rotated_pairs, const float theta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head = head_dim / 2;
    int total_q_pairs = seq_len * num_heads * rotated_pairs;
    int total_k_pairs = seq_len * num_kv_heads * rotated_pairs;

    if (idx < total_q_pairs)
    {
        int pair = idx % rotated_pairs;
        int remainder = idx / rotated_pairs;
        int head = remainder % num_heads;
        int t = remainder / num_heads;

        // Frequency denominator is the FULL head dim (gemma4 freq_factors).
        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)head_dim);
        float angle = (float)positions[t] * freq;
        float cos_val = cosf(angle), sin_val = sinf(angle);

        int base_idx = t * num_heads * head_dim + head * head_dim;
        int i0 = base_idx + pair;
        int i1 = base_idx + pair + half_head;

        float v0 = q[i0], v1 = q[i1];
        q[i0] = v0 * cos_val - v1 * sin_val;
        q[i1] = v0 * sin_val + v1 * cos_val;
    }

    if (idx < total_k_pairs)
    {
        int pair = idx % rotated_pairs;
        int remainder = idx / rotated_pairs;
        int head = remainder % num_kv_heads;
        int t = remainder / num_kv_heads;

        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)head_dim);
        float angle = (float)positions[t] * freq;
        float cos_val = cosf(angle), sin_val = sinf(angle);

        int base_idx = t * num_kv_heads * head_dim + head * head_dim;
        int i0 = base_idx + pair;
        int i1 = base_idx + pair + half_head;

        float v0 = k[i0], v1 = k[i1];
        k[i0] = v0 * cos_val - v1 * sin_val;
        k[i1] = v0 * sin_val + v1 * cos_val;
    }
}

// ── In-place scalar multiply (layer_output_scale) ──
extern "C" __global__ void __launch_bounds__(256) scale_inplace_f32(
    float* __restrict__ x, const int n, const float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        x[idx] = x[idx] * scale;
}

// ── Weight-less per-row RMSNorm (unit gamma) ──
// output[row] = x[row] * rsqrt(mean(x^2) + eps). One block per row; row width n.
// Used for the gemma4 weight-less V-norm with one row per (token, kv-head).
extern "C" __global__ void __launch_bounds__(256) rmsnorm_weightless_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    float* y = output + (size_t)row * n;
    const int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
    {
        float v = x[i];
        sum_sq = fmaf(v, v, sum_sq);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    __shared__ float ws[32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) ws[wid] = sum_sq;
    __syncthreads();

    if (wid == 0)
    {
        int nw = (blockDim.x + 31) >> 5;
        sum_sq = (lane < nw) ? ws[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
        if (lane == 0)
            ws[0] = rsqrtf(fmaf(sum_sq, 1.0f / (float)n, eps));
    }
    __syncthreads();
    const float ri = ws[0];

    for (int i = tid; i < n; i += blockDim.x)
        y[i] = x[i] * ri;
}

// ── Gemma-4 top-k renorm with the 6.1e-5 denominator clamp ──
// row[i] *= 1 / max(sum, 6.103515625e-05). Mirrors the CPU Gemma4Moe renorm
// (gemma4.cpp): the denominator is clamped at 2^-14 to avoid dividing by a
// vanishing top-k mass. One block per token; thread 0 does the small reduction.
extern "C" __global__ void __launch_bounds__(32) moe_renorm_topk_clamped_f32(
    float* __restrict__ topk_weight,       // [seq_len, top_k]
    const int seq_len, const int top_k)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;
    if (threadIdx.x != 0) return;

    float* row = topk_weight + (size_t)t * top_k;
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) sum += row[i];
    if (sum < 6.103515625e-05f) sum = 6.103515625e-05f;
    float inv = 1.0f / sum;
    for (int i = 0; i < top_k; i++) row[i] *= inv;
}

// ── Final-logit soft-capping (in place): x = c * tanh(x / c) ──
extern "C" __global__ void __launch_bounds__(256) softcap_inplace_f32(
    float* __restrict__ x, const int n, const float cap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float inv = 1.0f / cap;
        x[idx] = cap * tanhf(x[idx] * inv);
    }
}
