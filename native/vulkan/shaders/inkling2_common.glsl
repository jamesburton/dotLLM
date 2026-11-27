// Shared helpers of the Inkling 2 (Thinking Machines, inkling2_loop) Vulkan kernels (issue #2034, epic #2025). Included after `#version 450` and
// GL_GOOGLE_include_directive. The semantics of every kernel are those of the CPU reference (Inkling2Ops / Inkling2TransformerModel, docs/INKLING2.md).
//
// Activation storage is BF16 (raw 16-bit words, see olala_common.glsl). Every kernel accumulates and reduces in F32 registers / shared memory and
// rounds to BF16 once, on the store to a global buffer. Defining ACT_F32 builds the F32-activation-storage REFERENCE arm (a measurement control that
// is compared against the CPU reference without llama.cpp's noise; never the default). The KV cache (F16 or q8_0), the short-conv state (BF16),
// and every F32 weight vector (norm gains, queries, clips, conv taps, router) keep their stored types in both arms.
#include "olala_common.glsl"

// tanh with a clamped argument: the argument is finite everywhere on the SITU / controller paths, but an unguarded tanh may overflow to NaN for |x| >~ 44.
float tanhF(float x) { return tanh(clamp(x, -20.0, 20.0)); }

// Workgroup sum (every thread receives the total). Up to 32 subgroups. Ends with a barrier so it can be called repeatedly.
shared float i2Part[32];
float wgSum(float v) {
    v = subgroupAdd(v);
    if (subgroupElect()) i2Part[gl_SubgroupID] = v;
    barrier();
    float t = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; ++i) t += i2Part[i];
    barrier();
    return t;
}
