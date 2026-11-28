// Interleaved M-RoPE (ggml IMROPE) for FP32 Q/K, NeoX pairing (j, j + rope_dim/2) (issue #2073, Qwen3.5 / Qwen3-VL image input).
//
// Same arithmetic as rope_f32.cu's NeoX branch (rope_type 1), except the position of a rotary pair is picked per pair
// from three per-row axes: pos3 = [3][seq_len] ints (t, y, x), and axis_per_pair[j] in {0 = t, 1 = y, 2 = x, 3 = none -> 0}
// (DotLLM.Core.Vision.MropePositions.AxisPerPair). angle = (float)pos_axis * freq, with freq the same per-pair inverse
// frequency rope_f32 uses (inv_freq[pair] for YaRN, else 1 / powf(theta, 2*pair/rope_dim)); with three equal axes the
// result is bit-identical to rope_f32 (same expressions, same precise-math build). Build WITHOUT --use_fast_math
// (it is not in build.sh / build_ptx.bat's fast-math list). The CPU oracle is RoPERotation.ApplyMrope.
//
// NOTE: the matching PTX is NOT committed from the Strix Halo dev host (no nvcc). A CUDA box regenerates it with
// native/build_ptx.bat (or the DotLLM.Cuda build, native/msbuild/CudaPtx.targets); until then CudaKernels.HasMropeF32 is false.

#include <math.h>

extern "C" __global__ void __launch_bounds__(256) rope_mrope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos3,                 // [3][seq_len]: t axis, y axis, x axis
    const unsigned char* __restrict__ axis_per_pair, // [rope_dim/2]
    const int seq_len, const int num_heads, const int num_kv_heads,
    const int head_dim, const int rope_dim, const float theta,
    const float* __restrict__ inv_freq,           // [rope_dim/2] YaRN inverse freqs, or nullptr
    const float mscale)                           // 1.0f when YaRN is inactive
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    if (idx < total_q_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_heads;
        int t = remainder / num_heads;

        int ax = axis_per_pair[pair];
        int p = ax < 3 ? pos3[ax * seq_len + t] : 0;
        float freq = (inv_freq != nullptr)
            ? inv_freq[pair]
            : 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)p * freq;
        float cos_val = cosf(angle) * mscale, sin_val = sinf(angle) * mscale;

        int base_idx = t * num_heads * head_dim + head * head_dim;
        int i0 = base_idx + pair;
        int i1 = base_idx + pair + half_rope;

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

        int ax = axis_per_pair[pair];
        int p = ax < 3 ? pos3[ax * seq_len + t] : 0;
        float freq = (inv_freq != nullptr)
            ? inv_freq[pair]
            : 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)p * freq;
        float cos_val = cosf(angle) * mscale, sin_val = sinf(angle) * mscale;

        int base_idx = t * num_kv_heads * head_dim + head * head_dim;
        int i0 = base_idx + pair;
        int i1 = base_idx + pair + half_rope;

        float v0 = k[i0], v1 = k[i1];
        k[i0] = v0 * cos_val - v1 * sin_val;
        k[i1] = v0 * sin_val + v1 * cos_val;
    }
}
