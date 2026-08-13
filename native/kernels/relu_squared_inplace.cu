// native/kernels/relu_squared_inplace.cu
//
// Plain elementwise squared-ReLU, single buffer, in place: x = max(0, x)^2.
// Bit-perfect-shape port of DotLLM.Cpu.Kernels.ReluSquared.Execute (used, un-gated, by
// NVIDIA Nemotron-H's FFN sub-layer — up -> relu_squared -> down, no gate). Distinct from
// relu2_f32.cu's relu2_f32/relu2glu_f32 (GLU-fused, two input buffers, BitNet MoE FFN) — this
// kernel takes exactly one buffer, matching ReluSquared.Execute's signature.

extern "C" __global__ void relu_squared_inplace_f32(float* __restrict__ x, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    v = v > 0.0f ? v : 0.0f;
    x[idx] = v * v;
}
