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

// ── Decode-time (seqLen==1) fused variant ───────────────────────────────────
// Issue #168: the general kernel above requires its caller to pre-concatenate
// [conv_state (d_conv-1 rows); new qkv row(s)] into one contiguous scratch
// buffer, then separately (a) SiLU-activate the output and (b) copy the
// trailing (d_conv-1) rows of that same scratch buffer back into conv_state
// for the next step — 3 cuMemcpyDtoDAsync launches + 1 silu_f32 launch
// bracketing every conv1d_causal_f32 launch, x48 GDN layers x every decode
// token. For the decode case (exactly one new row), all of that is pure
// data-movement around a single-channel-wide 4-tap window and can be folded
// into ONE kernel:
//   - reads conv_state and the new qkv row directly (no physical concat)
//   - writes SiLU(conv output) directly (folds in silu_f32)
//   - writes the shifted trailing state directly (folds in the state-update
//     memcpy) — new_state[j] = old_state[j+1] for j<d_conv-2, and
//     new_state[d_conv-2] = the new (pre-conv) qkv value
//
// Aliasing: `state` doubles as input (old rolling history) and output (new
// rolling history) and `qkv_out` may alias `qkv_in` (conv1d_causal.cu's
// caller already overwrites the same qkv buffer it read from). Safe because
// each thread owns exactly one channel `c` end-to-end: no other thread ever
// touches column c of state/qkv_in/qkv_out, and every read that could be
// clobbered by a later write in this same thread (the taps) is staged into
// registers before any write happens.
//
// Only handles the real d_conv==4 shape explicitly (as above) plus a small
// generic fallback for completeness; GDN_CONV1D_DECODE_MAX_TAPS bounds the
// per-thread register array for the generic path (d_conv is a model-config
// constant, never grown at runtime, and 4 is the only value seen in any
// GGUF this codebase supports today — see the file-level comment above).
#define GDN_CONV1D_DECODE_MAX_TAPS 8

extern "C" __global__ void __launch_bounds__(256) gdn_conv1d_causal_decode_f32(
    float* __restrict__ state,              // [(d_conv-1), channels], in AND out
    const float* __restrict__ qkv_in,       // [channels], the one new (pre-conv) row
    const float* __restrict__ weight,       // [d_conv, channels] channel-major, w(k,c) at c*d_conv+k
    const float* __restrict__ bias,         // [channels]
    float* __restrict__ qkv_out,            // [channels], SiLU(conv output); may alias qkv_in
    const int d_conv, const int channels)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const float* w = weight + (size_t)c * d_conv;
    float acc = bias[c];

    if (d_conv == 4)
    {
        // Same accumulation order as conv1d_causal_f32's own d_conv==4 branch:
        // bias, then k=0..3 in order, each tap = state[k] for k<3, else the new row.
        float s0 = state[(size_t)0 * channels + c];
        float s1 = state[(size_t)1 * channels + c];
        float s2 = state[(size_t)2 * channels + c];
        float qn = qkv_in[c];

        acc += s0 * w[0];
        acc += s1 * w[1];
        acc += s2 * w[2];
        acc += qn * w[3];

        qkv_out[c] = acc * (1.0f / (1.0f + expf(-acc)));

        // Shift left, insert the new (pre-conv) row at the tail — matches the
        // general path's "copy the last (d_conv-1) rows of the virtual
        // [state; qkv] buffer" exactly, for seq_len==1.
        state[(size_t)0 * channels + c] = s1;
        state[(size_t)1 * channels + c] = s2;
        state[(size_t)2 * channels + c] = qn;
        return;
    }

    // Generic path (any d_conv) — k order identical to conv1d_causal_f32's
    // generic path; taps[] staged in registers so the state shift below can't
    // read an already-overwritten value.
    float taps[GDN_CONV1D_DECODE_MAX_TAPS];
    for (int k = 0; k < d_conv; k++)
    {
        taps[k] = (k < d_conv - 1) ? state[(size_t)k * channels + c] : qkv_in[c];
        acc += taps[k] * w[k];
    }

    qkv_out[c] = acc * (1.0f / (1.0f + expf(-acc)));

    for (int j = 0; j < d_conv - 1; j++)
    {
        state[(size_t)j * channels + c] = taps[j + 1];
    }
}
