// Fused squared-ReLU GLU with FP32 data (BitNet b1.58 FFN gating).
// out[i] = relu(gate[i])^2 * up[i]

extern "C" __global__ void __launch_bounds__(256) relu2_f32(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, const int n, const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * seq_len)
    {
        float g = gate[idx], u = up[idx];
        float r = g > 0.0f ? g : 0.0f;
        output[idx] = r * r * u;
    }
}
