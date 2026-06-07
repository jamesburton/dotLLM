// Depthwise causal 1-D convolution (FP32). Bit-perfect port of
// DotLLM.Cpu.Kernels.Conv1dCausal.ExecuteScalar.
//
// Layouts (matches llama.cpp / GGUF exactly):
//   input  : [d_conv-1 + seq_len, channels]  row-major  (caller prepends conv_state)
//   weight : [d_conv, channels]              channel-major (GGUF):  w(k,c) at c*d_conv + k
//   bias   : [channels]                      (caller passes zeros when the model has no bias —
//                                             the add is unconditional)
//   output : [seq_len, channels]             row-major
//
// Per output element:
//   y[t, c] = bias[c] + sum_{k=0..d_conv-1} input[(t+k)*channels + c] * weight[c*d_conv + k]
//
// Parallelization: one thread per (t, c). The accumulation is single-threaded, so
// the float-add order matches the CPU reference bit-for-bit by construction.
//
// d_conv == 4 is the universal case for Qwen3MoeHybrid / Mamba2 GGUF — taps are
// hoisted into registers and the FMA chain is fully unrolled. Other d_conv values
// take a plain in-register loop.

extern "C" __global__ void __launch_bounds__(256) conv1d_causal_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int d_conv, const int channels, const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * channels;
    if (idx >= total) return;

    int t = idx / channels;
    int c = idx - t * channels;

    const float* w = weight + (size_t)c * d_conv;
    // input[(t+k), c] = input[(t+k)*channels + c]; advance by `channels` per k.
    const float* in_col = input + (size_t)t * channels + c;

    float acc = bias[c];

    if (d_conv == 4)
    {
        // Hoist the 4 taps into registers; CPU loop is k=0..3 in order, so we
        // accumulate in the same order to preserve float-add associativity.
        float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
        float x0 = in_col[0];
        float x1 = in_col[(size_t)1 * channels];
        float x2 = in_col[(size_t)2 * channels];
        float x3 = in_col[(size_t)3 * channels];
        acc += x0 * w0;
        acc += x1 * w1;
        acc += x2 * w2;
        acc += x3 * w3;
    }
    else
    {
        // Generic path — keep k order identical to the CPU reference.
        for (int k = 0; k < d_conv; k++)
        {
            acc += in_col[(size_t)k * channels] * w[k];
        }
    }

    output[(size_t)t * channels + c] = acc;
}
