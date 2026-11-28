// Shared helpers of the Olala (OVHcloud) Vulkan kernels (#1983). Included after `#version 450` and GL_GOOGLE_include_directive.
//
// Activation storage. Olala's activations reach 1e9..1e14 (docs/OLALA.md section 3), so F16 storage overflows; stored activations are BF16
// (raw 16-bit words, `uint16_t`), which keeps the F32 exponent range. All arithmetic, accumulation and reductions happen in F32 registers /
// shared memory; a value is rounded to BF16 exactly once, when it is stored to a global buffer. Defining ACT_F32 builds the F32-storage
// REFERENCE arm (a test/measurement control for the precision sign-off; never selected by default).
//
// Recurrent state, the F32 gamma/bias vectors and the logits are always F32 (OlalaStatePolicy). The attention KV cache is always BF16.

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#ifdef ACT_F32
#define ACT_T float
float actToF(float v) { return v; }
float fToAct(float v) { return v; }
#elif defined(ACT_F16)
// F16-activation-storage MEASUREMENT arm (Inkling 2 activation-precision study, epic #2025): IEEE half, round to nearest even, overflow -> +-inf (NOT saturated), so a
// run that overflows is visible. Never selected by default; Olala must not use it (its activations reach 1e9..1e14).
#define ACT_T uint16_t
float actToF(uint16_t v) { return unpackHalf2x16(uint(v)).x; }
uint16_t fToAct(float f) {
    uint b = floatBitsToUint(f);
    uint sg = (b >> 16) & 0x8000u;
    uint a = b & 0x7fffffffu;
    if (a > 0x7f800000u) return uint16_t(sg | 0x7e00u);                  // NaN
    if (a >= 0x477ff000u) return uint16_t(sg | 0x7c00u);                 // >= 65520 rounds to inf (also +-inf)
    if (a < 0x38800000u) return uint16_t(sg | uint(roundEven(abs(f) * 16777216.0)));   // subnormal half: k * 2^-24; a carry into 0x400 is the smallest normal
    uint r = a + 0xfffu + ((a >> 13u) & 1u);
    return uint16_t(sg | ((r - 0x38000000u) >> 13u));
}
#else
#define ACT_T uint16_t
float actToF(uint16_t v) { return uintBitsToFloat(uint(v) << 16); }
// Round to nearest even; NaN stays NaN.
uint16_t fToAct(float f) {
    uint b = floatBitsToUint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return uint16_t((b >> 16) | 0x40u);
    b += 0x7fffu + ((b >> 16) & 1u);
    return uint16_t(b >> 16);
}
#endif

float bf16ToF(uint16_t v) { return uintBitsToFloat(uint(v) << 16); }
float bf16Lo(uint w) { return uintBitsToFloat(w << 16); }
float bf16Hi(uint w) { return uintBitsToFloat(w & 0xFFFF0000u); }
// Two packed activations of a 32-bit word (low / high half): the vectorised x loads of the GEMV / MoE kernels.
#ifdef ACT_F16
float actWordLo(uint w) { return unpackHalf2x16(w).x; }
float actWordHi(uint w) { return unpackHalf2x16(w).y; }
#else
#define actWordLo bf16Lo   // the very functions of the BF16 weight decode: the BF16 arm's SPIR-V stays byte-identical
#define actWordHi bf16Hi
#endif
// Round to nearest even BF16 bits (KV cache is always BF16, in both activation arms).
uint16_t fToBf16(float f) {
    uint b = floatBitsToUint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return uint16_t((b >> 16) | 0x40u);
    b += 0x7fffu + ((b >> 16) & 1u);
    return uint16_t(b >> 16);
}

float sigmoidF(float x) { return 1.0 / (1.0 + exp(-x)); }
float siluF(float x) { return x / (1.0 + exp(-x)); }

// log(1 + u) for u in [0, 1] without the cancellation of log(1.0 + u) (Kahan): GLSL has no log1p.
float log1pF(float u) {
    float w = 1.0 + u;
    return (w == 1.0) ? u : log(w) * (u / (w - 1.0));
}
// softplus with PyTorch's linear threshold at 20 (the CPU evaluates it in double; this is the F32 equivalent without losing the tail).
float softplusF(float x) {
    if (x > 20.0) return x;
    float u = exp(-abs(x));
    return max(x, 0.0) + log1pF(u);
}
