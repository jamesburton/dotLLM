// Fused squared-ReLU GLU activation kernel for dotLLM (BitNet b1.58 FFN gating).
// out[i] = relu(gate[i])^2 * up[i] = max(0, gate[i])^2 * up[i]
// Mirrors swiglu.cu: half2 loads/stores, FP32 computation, odd-tail handling.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) relu2_f16(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    const int n,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * seq_len;
    int total2 = total / 2;

    if (idx < total2)
    {
        half2 g2 = reinterpret_cast<const half2*>(gate)[idx];
        half2 u2 = reinterpret_cast<const half2*>(up)[idx];

        float g0 = __low2float(g2), g1 = __high2float(g2);
        float u0 = __low2float(u2), u1 = __high2float(u2);

        // relu(g)^2 * u
        float r0 = g0 > 0.0f ? g0 : 0.0f;
        float r1 = g1 > 0.0f ? g1 : 0.0f;
        float s0 = r0 * r0 * u0;
        float s1 = r1 * r1 * u1;

        reinterpret_cast<half2*>(output)[idx] = __floats2half2_rn(s0, s1);
    }

    // Handle odd trailing element
    if ((total & 1) && idx == 0)
    {
        int last = total - 1;
        float g = __half2float(gate[last]);
        float u = __half2float(up[last]);
        float r = g > 0.0f ? g : 0.0f;
        output[last] = __float2half(r * r * u);
    }
}
