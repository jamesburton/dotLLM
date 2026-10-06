// RoPE kernel with FP32 Q/K data.
// Q and K are processed independently — reuse attempts hurt GQA models.
//
// YaRN scaling (#366): when `inv_freq` is non-null it supplies the ramped per-pair
// inverse frequencies (rope_dim/2 floats, uploaded once at load from the CPU
// reference RoPE.ComputeYarnInverseFrequencies) and replaces the in-kernel
// powf(theta, ...). `mscale` is YaRN's attention-magnitude concentration factor,
// multiplied into BOTH cos and sin at EVERY position — including position 0.
// Non-YaRN callers pass inv_freq = nullptr and mscale = 1.0f, which is bit-identical
// to the previous behaviour (IEEE multiply by exactly 1.0f is the identity).
// NOTE: inv_freq already encodes the frequency denominator, so it supersedes
// `freq_dim`; the two are never combined (Gemma-4 partial-rotary uses freq_dim and
// has no YaRN scaling, dense YaRN models use inv_freq and pass freq_dim = 0).

#include <math.h>

extern "C" __global__ void __launch_bounds__(256) rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    const int seq_len, const int num_heads, const int num_kv_heads,
    const int head_dim, const int rope_dim, const float theta, const int rope_type,
    const int freq_dim, const int neox_pair_offset_arg,
    const float* __restrict__ inv_freq,  // [rope_dim/2] YaRN inverse freqs, or nullptr
    const float mscale)                  // YaRN cos/sin multiplier; 1.0f when inactive
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    // NeoX rotate-half pairing offset. STANDARD partial-rotary (Qwen3 / NemotronH /
    // Llama-family) pairs WITHIN the rotated block → rope_dim/2 (matches CPU
    // RoPE.Execute → ApplyRotationNeoX and native rope.cu / fused_rope_kv_write.cu).
    // Gemma-4 partial-rotary global layers pair ACROSS the full-head halves →
    // head_dim/2 (matches CPU RoPE.ApplyRotationNeoXPartial); the caller supplies that
    // via neox_pair_offset_arg. A prior change hardcoded head_dim/2 here, which
    // silently broke every other partial-rotary NeoX model (surfaced on Vulkan/CPU via
    // the Qwen3MoeHybrid IQ3 forward parity failure). For full rope both coincide.
    // Callers pass 0 to mean the standard rope_dim/2. NOTE: PTX must be regenerated on
    // the CUDA box (no nvcc on the Strix Halo dev host).
    int neox_pair_offset = neox_pair_offset_arg > 0 ? neox_pair_offset_arg : half_rope;
    // Frequency-denominator dim for the exponent 2*pair/freq_dim. Equals rope_dim for
    // full rotation AND standard partial NeoX; for Gemma-4 partial global freq_dim is
    // the FULL head dim, matching the CPU oracle's partial freq table. 0 ⇒ rope_dim.
    int fd = freq_dim > 0 ? freq_dim : rope_dim;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    if (idx < total_q_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_heads;
        int t = remainder / num_heads;

        float freq = (inv_freq != nullptr)
            ? inv_freq[pair]
            : 1.0f / powf(theta, (float)(2 * pair) / (float)fd);
        float angle = (float)positions[t] * freq;
        float cos_val = cosf(angle) * mscale, sin_val = sinf(angle) * mscale;

        int base_idx = t * num_heads * head_dim + head * head_dim;
        int i0 = (rope_type == 1) ? base_idx + pair : base_idx + 2 * pair;
        int i1 = (rope_type == 1) ? base_idx + pair + neox_pair_offset : base_idx + 2 * pair + 1;

        float v0 = q[i0], v1 = q[i1];
        q[i0] = v0 * cos_val - v1 * sin_val;
        q[i1] = v0 * sin_val + v1 * cos_val;
    }

    if (idx < total_k_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_kv_heads;
        int t = remainder / num_kv_heads;

        float freq = (inv_freq != nullptr)
            ? inv_freq[pair]
            : 1.0f / powf(theta, (float)(2 * pair) / (float)fd);
        float angle = (float)positions[t] * freq;
        float cos_val = cosf(angle) * mscale, sin_val = sinf(angle) * mscale;

        int base_idx = t * num_kv_heads * head_dim + head * head_dim;
        int i0 = (rope_type == 1) ? base_idx + pair : base_idx + 2 * pair;
        int i1 = (rope_type == 1) ? base_idx + pair + neox_pair_offset : base_idx + 2 * pair + 1;

        float v0 = k[i0], v1 = k[i1];
        k[i0] = v0 * cos_val - v1 * sin_val;
        k[i1] = v0 * sin_val + v1 * cos_val;
    }
}
